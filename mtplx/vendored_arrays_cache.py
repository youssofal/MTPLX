"""Vendored ``ArraysCache`` advance() Metal buffer-object leak fix.

Vendored from mlx-lm PR #1642 @ 985af30 by thinkroth (Apache-2.0, same
license as upstream mlx-lm; see the repository NOTICE file). This module
ports the fixed ``ArraysCache`` from that PR's ``mlx_lm/models/cache.py``
as ``FixedArraysCache``, a subclass of the installed stock class, so that
fresh installs resolving stock mlx-lm 0.31.x (whose ``advance()`` builds
one dead lazy-graph node per layer per token until Metal's buffer-object
count limit kills the process) are safe without a git-pinned environment.

Subclassing — rather than replacing — the stock class keeps every
``isinstance(entry, ArraysCache)`` check in this repo (e.g. graphbank's
verify-state spec builder) working for both patched and unpatched caches.

Installation is handled by :mod:`mtplx.arrays_cache_patch`, which swaps
``mlx_lm.models.cache.ArraysCache`` for this class only when the probe
shows upstream lacks the fix. Against an upstream that already carries
the fix natively this module stays importable and inert (the installer
never binds it).

Faithfulness: semantics, code, and comments are ported verbatim from the
pinned implementation. The only adaptations are:

1. ``FixedArraysCache`` subclasses the live ``mlx_lm.models.cache
   .ArraysCache`` (upstream's class subclasses ``_BaseCache``); members
   whose upstream fixed body is textually identical to stock 0.31.x are
   inherited rather than duplicated: ``__init__``, ``__setitem__``,
   ``__getitem__``, ``state``, ``prepare``, ``merge``, ``empty``,
   ``nbytes`` and ``_BaseCache.from_state``.
2. ``__new__`` bypasses the stock ``__new__`` body with
   ``object.__new__`` (see the inline note) and reproduces its effect
   through the private slots.
3. ``extract()`` constructs ``FixedArraysCache`` where upstream names the
   module-global ``ArraysCache`` (which in the pinned tree IS the fixed
   class).
4. The module-level helpers and the weakref array registry live here
   instead of inside ``mlx_lm.models.cache``.
"""

import copy
import operator
import threading
import weakref
from contextlib import ExitStack, contextmanager

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache as _StockArraysCache


def _try_schedule(*arrays):
    """Schedule an async evaluation unless it is disallowed: inside a
    graph transformation (``mx.compile`` traces captured state by
    temporarily swapping tracers into the captured containers, e.g.
    ``inputs=vars(cache)``) ``async_eval`` on a tracer raises, and a
    metadata access must instead behave like stock's plain attribute
    read -- return the tracer, schedule nothing, and leave any pending
    fold for the next eager access. Returns False in that case."""
    try:
        mx.async_eval(*arrays)
        return True
    except ValueError as e:
        if "graph transformation" not in str(e):
            raise
        return False


class _ArraysCacheFoldLock(type(threading.RLock())):
    """An RLock used by a cache alias group or a metadata-array alias group.

    Normal object sharing keeps one lock across aliases. Pickle and deepcopy
    create a new native lock; their memo preserves sharing when an alias graph
    contains the same lock more than once.
    """

    def __reduce__(self):
        return (type(self), ())


class _ArraysCacheArraySync:
    """Synchronization and physical-promotion state for one backing array."""

    def __init__(self, promotion=None, lock=None):
        self.promotion = [False] if promotion is None else promotion
        self.lock = _ArraysCacheFoldLock() if lock is None else lock


# Public setters can attach one mx.array to independently constructed caches.
# Keep synchronization by backing-array identity so their deferred reads cannot
# concurrently mutate that array on different thread-local MLX streams. The
# weak registry does not retain arrays after their last real owner is gone.
_ARRAYS_CACHE_ARRAY_SYNCS = {}
_ARRAYS_CACHE_ARRAY_SYNCS_LOCK = threading.Lock()


def _register_arrays_cache_array(arr, state):
    if arr is None:
        return state
    if not hasattr(state, "array_lock"):
        # Compatibility with cache pickles produced before array-level locks.
        state.array_lock = _ArraysCacheFoldLock()
    key = id(arr)
    with _ARRAYS_CACHE_ARRAY_SYNCS_LOCK:
        item = _ARRAYS_CACHE_ARRAY_SYNCS.get(key)
        if item is not None and item[0]() is arr:
            sync = item[1]
            state.promotion = sync.promotion
            state.array_lock = sync.lock
            return state

        sync = _ArraysCacheArraySync(state.promotion, state.array_lock)

        def remove(ref, key=key):
            with _ARRAYS_CACHE_ARRAY_SYNCS_LOCK:
                current = _ARRAYS_CACHE_ARRAY_SYNCS.get(key)
                if current is not None and current[0] is ref:
                    del _ARRAYS_CACHE_ARRAY_SYNCS[key]

        ref = weakref.ref(arr, remove)
        _ARRAYS_CACHE_ARRAY_SYNCS[key] = (ref, sync)
        return state


def _arrays_cache_field_state(arr=None, advance=0, promoted=False):
    return _register_arrays_cache_array(
        arr, _ArraysCacheFieldState(advance=advance, promoted=promoted)
    )


class _ArraysCacheFieldState:
    """Mutable bookkeeping shared by shallow aliases of one field.

    ``promotion`` is separate from the pending counter because the two
    metadata fields can reference the same bool array: stock promoted
    both logical views together, while each field still applied its own
    decrement.
    """

    def __init__(self, advance=0, promoted=False, promotion=None, array_lock=None):
        self.advance = advance
        self.promotion = [promoted] if promotion is None else promotion
        self.array_lock = _ArraysCacheFoldLock() if array_lock is None else array_lock


class FixedArraysCache(_StockArraysCache):
    # ``left_padding`` and ``lengths`` are logically decremented by
    # ``advance()`` on every layer call during decode. Doing that decrement
    # with mx.array arithmetic (``self.left_padding -= N``) builds one
    # unevaluated graph node per layer per token. The model only ever
    # rebuilds a mask from ONE of these caches (``cache[ssm_idx]``), so for
    # every other layer the chain is dead: it is never evaluated, it grows
    # without bound, and every link pins the small constant array (and its
    # live Metal buffer object) that was subtracted. The process then dies
    # with ``[metal::malloc] Resource limit (499000) exceeded`` -- a count
    # limit, not bytes -- after a few tens of thousands of decode tokens
    # (see ml-explore/mlx-lm#1185, #1332; ml-explore/mlx#3564, #3539).
    #
    # Fix: track the cumulative decrement as a plain python int per field
    # (``_lp_advance``/``_len_advance``; nonzero only while the matching
    # array exists) and fold it into the stored array when that field is
    # read (property access or ``make_mask``), replaced, or finalized.
    # ``advance()`` itself creates no graph nodes and never evaluates
    # anything; folding one field never touches the other. Folds are
    # applied in place (``-=``, preserving the array object) on the
    # metadata side and scheduled with ``mx.async_eval``, so no public
    # operation sequence accumulates unevaluated graph on metadata
    # nothing consumes.
    #
    # Deliberate deviations from the eager decrement (nothing in this
    # repo relies on them): a decrement becomes visible to aliases of a
    # metadata array at the next fold of that field, not at ``advance()``
    # time -- alias reads/writes before that fold interleave accordingly,
    # and storing the same array object in both fields applies each
    # field's decrement at its own fold; ``copy.copy`` folds first, so a
    # shallow copy cannot re-apply pending decrements to the shared
    # arrays. Deferred totals fold as the dtype-congruent scalar (MLX
    # converts python int scalars through int64, so uint64 uses the
    # signed representative modulo 2**64), which lands integer metadata
    # exactly where stock's per-step subtractions did -- including
    # totals that overflow the dtype. Offsets outside the scalar range
    # MLX accepts for the dtype raise a uniform ValueError at
    # ``advance()``, where stock's eager subtraction rejected them (as
    # ValueError, or an opaque std::bad_cast at the int64 gate).
    # Floating totals beyond the int64 gate fold in gate-sized chunks,
    # each scheduled as it is applied; per-step float *rounding* is not
    # reproduced. ``operator.index`` normalizes integral offsets
    # (python bool and numpy integer scalars become python ints, value
    # preserved) -- stock instead converted exotic offset types through
    # array promotion, whose dtype side effects (a bool offset keeping
    # bool metadata bool, numpy scalar offsets promoting the metadata
    # dtype) are not reproduced. bool metadata arithmetic is reproduced
    # at int32 precision (stock's promotion): the dtype stays bool
    # through zero-net advance histories. Physical promotion and fold
    # synchronization follow the backing array even when public setters
    # attach it to independently constructed caches; shallow aliases
    # additionally share their per-field pending counter. After the first
    # advance() on a bool field -- even a zero or cancelling one -- later
    # offsets through any of those aliases validate against int32 exactly
    # as stock's already-promoted array would. A read that encounters a *tracer*
    # -- metadata swapped into a trace by
    # ``mx.compile(..., inputs=vars(cache))`` -- is pure: no fold, no
    # scheduling, like stock's attribute access, and the pending
    # decrement folds at the next eager access; a closure-captured
    # cache holds concrete arrays, so reads inside a trace fold eagerly
    # with stock-identical values. Compiling a function that *calls*
    # ``advance()`` is not supported: the decrement is host-side
    # bookkeeping and runs at trace time only (stock incidentally
    # recorded it as graph arithmetic -- the same arithmetic that
    # leaks), so call advance() outside compiled functions. Likewise,
    # mutating metadata (assignment, ``finalize``, ``filter``,
    # ``extend``) while a pending decrement cannot fold -- the field is
    # captured as a tracer -- raises rather than silently discarding
    # the decrement (``copy.copy`` and ``copy.deepcopy`` guard the same
    # way -- a copy taken mid-trace would escape holding the transient
    # tracer); mutate outside compiled functions. In-place item
    # assignment through a captured tracer read cannot be intercepted
    # (``__setitem__`` on the returned array bypasses the property
    # machinery): the write lands before the deferred decrement -- the
    # alias-write window above -- where stock's eager decrement landed
    # first. Reads are serialized per cache: a reentrant lock guards the
    # advance bookkeeping, folds, scheduling, copies, and metadata
    # serialization, because stock's pure attribute reads were trivially
    # thread-safe while unserialized concurrent folds of one array
    # deadlock on their thread-local streams. A second, array-identity
    # lock serializes independently constructed caches that share a
    # backing array. Shallow aliases share their cache lock and per-field
    # bookkeeping; deepcopy preserves that topology within an alias
    # graph. Quiescent generic pickle preserves the same topology, but
    # pickle does not expose a lifecycle hook that can hold the source
    # lock for the complete outer Pickler traversal: concurrent generic
    # pickle is unsupported. ``meta_state`` and ``save_prompt_cache``
    # remain linearizable complete metadata snapshots.
    # Cross-thread *mutation* (filter/extend/finalize racing anything)
    # remains unsupported, as in stock.

    def __new__(cls, *args, **kwargs):
        # Vendoring note: upstream's fixed __new__ calls
        # ``super().__new__(cls)`` onto a base that assigns nothing. Here
        # the base is the STOCK ArraysCache, whose __new__ body assigns
        # ``instance.left_padding = None`` / ``instance.lengths = None``
        # as plain attributes -- on this class those names are data
        # descriptors whose setters need the private state to exist
        # first. Bypass that body with object.__new__ (the stock base
        # chain adds nothing else) and reproduce its effect through the
        # private slots the properties read.
        instance = object.__new__(cls)
        instance._left_padding = None
        instance._lengths = None
        instance._lp_state = _ArraysCacheFieldState()
        instance._len_state = _ArraysCacheFieldState()
        # bool-metadata promotion trackers: stock's eager subtraction
        # promoted a bool field to int32 on the FIRST advance() -- even a
        # zero or cancelling one -- so validation must remember that
        # independently of the pending total (see _logical_dtype)
        # Serializes the deferred-decrement machinery (advance
        # bookkeeping, folds, scheduling, copies, serialization). The
        # native lock subclass is pickleable and deepcopy-aware without
        # adding a wrapper on the per-token advance() hot path.
        instance._fold_lock = _ArraysCacheFoldLock()
        return instance

    @property
    def _lp_advance(self):
        return self._lp_state.advance

    @_lp_advance.setter
    def _lp_advance(self, value):
        self._lp_state.advance = value

    @property
    def _len_advance(self):
        return self._len_state.advance

    @_len_advance.setter
    def _len_advance(self, value):
        self._len_state.advance = value

    @property
    def _lp_promoted(self):
        return self._lp_state.promotion[0]

    @_lp_promoted.setter
    def _lp_promoted(self, value):
        self._lp_state.promotion[0] = value

    @property
    def _len_promoted(self):
        return self._len_state.promotion[0]

    @_len_promoted.setter
    def _len_promoted(self, value):
        self._len_state.promotion[0] = value

    # Integer metadata dtypes as (bits, signed): used to mirror stock's
    # eager per-call scalar range check in advance() and to wrap the
    # accumulated total to the dtype's modular range at fold time.
    _INT_DTYPES = {
        "int8": (8, True),
        "int16": (16, True),
        "int32": (32, True),
        "int64": (64, True),
        "uint8": (8, False),
        "uint16": (16, False),
        "uint32": (32, False),
        "uint64": (64, False),
    }

    @classmethod
    def _int_spec(cls, dtype):
        return cls._INT_DTYPES.get(str(dtype).split(".")[-1])

    @classmethod
    def _wrap_advance(cls, total, dtype):
        """Wrap an accumulated advance() total to the congruent scalar
        MLX converts exactly like stock's per-step decrements did.
        Sequential wrapping subtractions equal one subtraction of the
        sum modulo 2**bits, so the wrapped fold lands on stock's value
        even when the raw total overflows the dtype. Python int scalars
        convert through int64 (mlx python/src/utils.cpp), so uint64
        takes the *signed* int64 representative -- congruent modulo
        2**64 and accepted by the conversion. Non-integer dtypes pass
        through (the fold subtracts them in gate-sized chunks). bool
        metadata wraps at int32: stock's subtraction promoted it, and
        int32 wrapping matches the promoted arithmetic (including the
        two's-complement truncation MLX's C++ cast applies to
        out-of-range scalars)."""
        if dtype == mx.bool_:
            spec = (32, True)
        else:
            spec = cls._int_spec(dtype)
        if spec is None:
            return total
        bits, signed = spec
        total %= 1 << bits
        if (signed or bits == 64) and total >= 1 << (bits - 1):
            total -= 1 << bits
        return total

    @classmethod
    def _check_advance(cls, N, dtype):
        """Mirror stock's eager scalar conversion: python ints convert
        through int64, so every dtype rejects offsets outside int64's
        range, and integer dtypes narrower than 64 bits are additionally
        range-checked (uint64 is not: negatives wrap modulo 2**64).
        Runs per armed field, in stock's field order, because that is
        where stock's ``-= N`` raised (ValueError, or std::bad_cast at
        the int64 gate; here it is uniformly ValueError)."""
        lo, hi = -(1 << 63), (1 << 63) - 1
        spec = cls._int_spec(dtype)
        if spec is not None and spec[0] < 64:
            bits, signed = spec
            lo = -(1 << (bits - 1)) if signed else 0
            hi = ((1 << (bits - 1)) if signed else (1 << bits)) - 1
        if not lo <= N <= hi:
            raise ValueError(
                f"ArraysCache.advance offset {N} is out of range "
                f"for {dtype} metadata"
            )

    def _fold(self, arr_attr, state_attr):
        """Fold the pending decrement into the backing array -- in place,
        so aliases observe it -- and schedule the evaluation. A no-op on
        tracers (see _try_schedule): the fold stays pending, the counter
        untouched. Integer totals fold as one pre-wrapped exact scalar;
        other dtypes subtract in int64-gate-sized chunks (an accumulated
        floating total can exceed the scalar range even when every
        offset was valid), each chunk scheduled as it is applied so a
        many-chunk fold cannot itself build an unbounded chain. The
        counter commits chunk-by-chunk, BEFORE each subtraction: a
        decrement is never applied twice, and a failure mid-fold leaves
        the unapplied remainder pending except for at most the single
        in-flight chunk (an asynchronous interrupt or failed subtraction
        between the counter commit and the subtraction drops it)."""
        with self._fold_lock:
            state = getattr(self, state_attr)
            with state.array_lock:
                total = state.advance
                if not total:
                    return
                arr = getattr(self, arr_attr)
                if not _try_schedule(arr):
                    return
                total = self._wrap_advance(total, arr.dtype)
                state.advance = total
                lo, hi = -(1 << 63), (1 << 63) - 1
                while True:
                    chunk = min(max(total, lo), hi)
                    total -= chunk
                    state.advance = total
                    arr -= chunk
                    setattr(self, arr_attr, arr)
                    if not total:
                        break
                    mx.async_eval(arr)
                mx.async_eval(arr)

    def _fold_lp(self):
        self._fold("_left_padding", "_lp_state")

    def _fold_len(self):
        self._fold("_lengths", "_len_state")

    @property
    def left_padding(self):
        if self._left_padding is None:
            return None
        with self._fold_lock:
            with self._lp_state.array_lock:
                if self._lp_advance:
                    self._fold_lp()
                else:
                    # Schedule on every access: external in-place writes
                    # through the returned array are otherwise never
                    # evaluated for metadata nothing consumes (a property
                    # access is not an evaluation).
                    _try_schedule(self._left_padding)
                return self._left_padding

    @left_padding.setter
    def left_padding(self, v):
        # Fold the outgoing array first so earlier aliases still observe
        # the decrement; otherwise replacement would discard it forever.
        with self._fold_lock:
            if self._left_padding is not None:
                self._fold_lp()
                if self._lp_advance:
                    # The fold declined: the outgoing array is a tracer
                    # (captured by a graph transformation). Proceeding
                    # would silently discard the decrement.
                    raise RuntimeError(
                        "ArraysCache metadata cannot be replaced inside "
                        "a graph transformation while an advance() "
                        "decrement is pending"
                    )
            same = v is self._left_padding
            state = self._lp_state if same else _arrays_cache_field_state(v)
            if v is not None:
                # Bound chains from repeatedly assigning lazy expressions
                with state.array_lock:
                    _try_schedule(v)
            self._left_padding = v
            if not same:
                # Only a genuinely new array resets the bool-promotion
                # record; re-assigning the backing array is not a reset
                self._lp_state = state
            else:
                self._lp_advance = 0

    @property
    def lengths(self):
        if self._lengths is None:
            return None
        with self._fold_lock:
            with self._len_state.array_lock:
                if self._len_advance:
                    self._fold_len()
                else:
                    _try_schedule(self._lengths)
                return self._lengths

    @lengths.setter
    def lengths(self, v):
        with self._fold_lock:
            if self._lengths is not None:
                self._fold_len()
                if self._len_advance:
                    raise RuntimeError(
                        "ArraysCache metadata cannot be replaced inside "
                        "a graph transformation while an advance() "
                        "decrement is pending"
                    )
            same = v is self._lengths
            state = self._len_state if same else _arrays_cache_field_state(v)
            if v is not None:
                with state.array_lock:
                    _try_schedule(v)
            self._lengths = v
            if not same:
                self._len_state = state
            else:
                self._len_advance = 0

    def _sync(self):
        """Fold both pending integer decrements into the stored arrays."""
        self._fold_lp()
        self._fold_len()

    @contextmanager
    def _metadata_locked(self):
        """Lock this cache and every distinct backing metadata array."""
        with self._fold_lock:
            locks = {}
            if self._left_padding is not None:
                locks[id(self._lp_state.array_lock)] = self._lp_state.array_lock
            if self._lengths is not None:
                locks[id(self._len_state.array_lock)] = self._len_state.array_lock
            with ExitStack() as stack:
                for key in sorted(locks):
                    stack.enter_context(locks[key])
                yield

    def _register_metadata_arrays(self):
        if self._left_padding is not None:
            self._lp_state = _register_arrays_cache_array(
                self._left_padding, self._lp_state
            )
        if self._lengths is not None:
            self._len_state = _register_arrays_cache_array(
                self._lengths, self._len_state
            )

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2:
            state, slot_state = state
        else:
            slot_state = None
        if state is not None:
            self.__dict__.update(state)
        if slot_state is not None:
            for key, value in slot_state.items():
                setattr(self, key, value)
        self._register_metadata_arrays()

    def _require_folded(self, op):
        """Guard for mutators: a pending counter after _sync() means the
        backing array is a tracer (captured by a graph transformation),
        and rebinding or clearing the field now would silently discard
        the decrement. Reads stay supported under capture; mutations
        must run eagerly. Never fires on eager paths, where folds
        always land."""
        if self._lp_advance or self._len_advance:
            raise RuntimeError(
                f"ArraysCache.{op}: metadata is captured by a graph "
                "transformation with an advance() decrement pending; "
                "apply cache mutations outside compiled functions"
            )

    def __copy__(self):
        # A shallow copy shares the backing arrays (as stock's did) --
        # plus the fold lock and mutable per-field bookkeeping, so a
        # pending total is folded only once across the alias graph.
        # Fold first to preserve stock's copy-time visibility. If the
        # fold cannot land (captured tracer), a copy would escape the
        # trace holding the transient tracer -- raise like the other
        # guarded mutations.
        with self._metadata_locked():
            self._sync()
            self._require_folded("__copy__")
            reducer = self.__reduce_ex__(4)
            if isinstance(reducer, str):
                return self
            new = copy._reconstruct(self, None, *reducer)
            new._register_metadata_arrays()
            return new

    def __deepcopy__(self, memo):
        # Same guard as __copy__: python's default deepcopy would copy
        # __dict__ directly, letting a deep copy taken mid-trace escape
        # with the transient tracer and the pending counter. The copy
        # gets an independent lock for an independent array, while a
        # shared deepcopy memo preserves lock/state sharing within a
        # copied shallow-alias graph.
        with self._metadata_locked():
            self._sync()
            self._require_folded("__deepcopy__")
            reducer = self.__reduce_ex__(4)
            if isinstance(reducer, str):
                return self
            new = copy._reconstruct(self, memo, *reducer)
            new._register_metadata_arrays()
            return new

    def _materialize(self):
        """Schedule evaluation of the metadata arrays. filter/extend run
        once per batch change, but only one layer's metadata is ever
        consumed downstream, so batch churn on a long-lived server would
        otherwise grow an unevaluated chain for every other layer (the
        same dead-graph pattern as advance(), at per-request rate).
        async_eval, not eval: a synchronous eval here would block behind
        whatever the generation loop already queued on the stream."""
        arrs = [a for a in (self._left_padding, self._lengths) if a is not None]
        if arrs:
            _try_schedule(arrs)

    @property
    def batch_size(self):
        for c in self.cache:
            if c is not None:
                return c.shape[0]
        if self._left_padding is not None:
            return self._left_padding.size
        elif self._lengths is not None:
            return self._lengths.size
        else:
            return 1

    # dtypes the metadata codec accepts, with the value parser for each.
    # Deliberately numeric-only: metadata holds padding/length counts.
    _META_DTYPES = {
        "int8": int,
        "int16": int,
        "int32": int,
        "int64": int,
        "uint8": int,
        "uint16": int,
        "uint32": int,
        "uint64": int,
        "float16": float,
        "float32": float,
        "float64": float,
        "bfloat16": float,
    }

    @property
    def meta_state(self):
        # Metadata is not part of ``state`` (its fields are optional and
        # ``state`` slots may not be None), so serialize it as strings:
        # "" for an absent field, "<dtype>:<comma-joined values>"
        # otherwise (an empty array stays distinct from None and dtype
        # survives the round trip).
        def encode(a):
            if a is None:
                return ""
            if a.ndim != 1:
                # The encoding is 1-D only -- the shape make_mask/merge/
                # filter arithmetic assumes. Fail loudly instead of
                # emitting an entry the decoder cannot parse.
                raise ValueError(
                    f"ArraysCache metadata must be 1-D to serialize, "
                    f"got shape {a.shape}"
                )
            dtype = str(a.dtype).split(".")[-1]
            if dtype not in self._META_DTYPES:
                raise TypeError(
                    f"ArraysCache metadata with dtype {a.dtype} cannot be serialized"
                )
            return dtype + ":" + ",".join(str(x) for x in a.tolist())

        with self._metadata_locked():
            if self._left_padding is None and self._lengths is None:
                return ""
            self._sync()
            return (encode(self._left_padding), encode(self._lengths))

    @meta_state.setter
    def meta_state(self, v):
        # Files written before metadata serialization carry an empty entry
        if not v:
            return

        def decode(s):
            if not s:
                return None
            dtype_name, sep, vals = s.partition(":")
            cast = self._META_DTYPES.get(dtype_name)
            if not sep or cast is None:
                raise ValueError(f"Malformed ArraysCache metadata entry: {s!r}")
            return mx.array(
                [cast(x) for x in vals.split(",")] if vals else [],
                dtype=getattr(mx, dtype_name),
            )

        lp, ln = v
        self.left_padding = decode(lp)
        self.lengths = decode(ln)

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        self.cache = [c[batch_indices] if c is not None else None for c in self.cache]
        self._sync()
        self._require_folded("filter")
        if self._left_padding is not None:
            promoted = self._lp_promoted
            new = self._left_padding[batch_indices]
            state = _arrays_cache_field_state(new, promoted=promoted)
            self._left_padding, self._lp_state = new, state
        if self._lengths is not None:
            promoted = self._len_promoted
            new = self._lengths[batch_indices]
            state = _arrays_cache_field_state(new, promoted=promoted)
            self._lengths, self._len_state = new, state
        self._materialize()

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """

        a_batch = self.batch_size
        b_batch = other.batch_size

        def cat(a, b):
            shape = dtype = None
            if a is not None:
                shape = a.shape
                dtype = a.dtype
            if b is not None:
                shape = b.shape
                dtype = b.dtype

            if shape is None:
                return None

            if a is None:
                a = mx.zeros((a_batch,) + shape[1:], dtype=dtype)
            if b is None:
                b = mx.zeros((b_batch,) + shape[1:], dtype=dtype)

            return mx.concatenate([a, b])

        def materialize_promoted_bool(a, promoted):
            # Stock's first subtraction physically promoted bool metadata
            # to int32. Preserve that logical dtype before mixed-dtype
            # concatenation (bool + int8 would otherwise become int8).
            if a is not None and promoted and a.dtype == mx.bool_:
                return a.astype(mx.int32)
            return a

        self._sync()
        other._sync()
        self._require_folded("extend")
        other._require_folded("extend")
        self.cache = [cat(c, o) for c, o in zip(self.cache, other.cache)]
        new = cat(
            materialize_promoted_bool(self._left_padding, self._lp_promoted),
            materialize_promoted_bool(other._left_padding, other._lp_promoted),
        )
        state = _arrays_cache_field_state(new)
        self._left_padding, self._lp_state = new, state
        new = cat(
            materialize_promoted_bool(self._lengths, self._len_promoted),
            materialize_promoted_bool(other._lengths, other._len_promoted),
        )
        state = _arrays_cache_field_state(new)
        self._lengths, self._len_state = new, state
        # Concatenation rebinds each field together with fresh bookkeeping as
        # soon as that field succeeds. If the second concatenation raises, the
        # first field therefore remains coherent and matches stock's partial
        # mutation order without retaining a shallow alias's pending counter.
        self._materialize()

    def extract(self, idx):
        cache = FixedArraysCache(len(self.cache))
        cache.cache = [c[idx : idx + 1] for c in self.cache]
        return cache

    def finalize(self):
        # Fold before clearing so aliases held by callers still observe
        # the decrements that happened while the fields were live
        self._sync()
        self._require_folded("finalize")
        self._lengths = None
        self._left_padding = None
        self._lp_state = _ArraysCacheFieldState()
        self._len_state = _ArraysCacheFieldState()

    def advance(self, N):
        # Integer bookkeeping only: building this with mx.array arithmetic
        # leaks one live buffer object per layer per token (see class note).
        # mx.array offsets are rejected loudly rather than coerced --
        # coercion would force an eval here, truncate floats, and accept
        # non-scalars that silently corrupt the deferred arithmetic.
        # operator.index accepts any integral python scalar and rejects
        # floats. Type validation runs regardless of whether any metadata
        # is set, so that contract does not depend on cache state; the
        # dtype range check below is per armed field, in stock's field
        # order, because that is exactly where stock's eager subtraction
        # would have rejected the offset.
        if isinstance(N, mx.array):
            raise TypeError("ArraysCache.advance requires a python int, not mx.array")
        N = operator.index(N)
        with self._fold_lock:
            if self._lengths is not None:
                promoted = self._len_promoted or (
                    self._lengths is self._left_padding and self._lp_promoted
                )
                self._check_advance(N, self._logical_dtype(self._lengths, promoted))
                self._len_advance += N
                if self._lengths.dtype == mx.bool_:
                    self._len_promoted = True
            if self._left_padding is not None:
                promoted = self._lp_promoted or (
                    self._left_padding is self._lengths and self._len_promoted
                )
                self._check_advance(
                    N, self._logical_dtype(self._left_padding, promoted)
                )
                self._lp_advance += N
                if self._left_padding.dtype == mx.bool_:
                    self._lp_promoted = True

    @staticmethod
    def _logical_dtype(arr, promoted):
        # Stock's eager subtraction promoted bool metadata to int32 on
        # the FIRST advance() -- even a zero or cancelling one, and in
        # place, so a bool array shared between both fields promoted
        # both (hence the identity checks at the call sites). With the
        # fold deferred, the stored dtype stays bool, so later offsets
        # must validate against the promoted dtype exactly as stock's
        # stored array would have.
        if promoted and arr.dtype == mx.bool_:
            return mx.int32
        return arr.dtype

    def make_mask(self, N: int):
        if self._left_padding is None and self._lengths is None:
            return None
        pos = mx.arange(N)
        # Both fields can be live at once: ``merge`` of fresh caches sets
        # left_padding to zeros and a ragged prefill then sets lengths. The
        # stock rule returned the left-padding mask alone in that case, so the
        # right-padded rows' pad tokens reached the recurrent update and
        # drifted the shorter rows' state (#420 follow-up, batched Flash-Next).
        # AND the two; fold (and schedule) only the fields the mask uses.
        mask = None
        with self._fold_lock:
            if self._left_padding is not None:
                with self._lp_state.array_lock:
                    self._fold_lp()
                    mask = pos >= self._left_padding[:, None]
            if self._lengths is not None:
                with self._len_state.array_lock:
                    self._fold_len()
                    len_mask = pos < self._lengths[:, None]
                    mask = len_mask if mask is None else (mask & len_mask)
        return mask
