"""Server entrypoints for MTPLX.

This package's ``__init__`` runs at exactly one useful moment: Python
executes it while resolving ``python -m mtplx.server.openai``, i.e. BEFORE
the first line of ``openai.py`` -- and therefore before that module's import
block pulls ``mtplx.runtime_options``, ``mtplx.generation``,
``mtplx.fable_block_verify`` and ``mtplx.fable_draft_k20_prescatter``, each
of which freezes several of the retained stack's env keys in a
module-level constant at ITS import.

The server's own retained-stack ``setdefault`` block does not run until
``mtplx/server/openai.py:_load``, thousands of imported lines later. For the
eleven ``BIND_IMPORT`` keys that is too late: the environment would change and
no reader would ever look again, so the server would report lanes it had
not actually armed. :func:`~mtplx.full_stack_env.stamp_import_time_defaults` closes
that window and nothing else -- it stamps only that subset, only for a served
Flash-Next pack (or an explicit ``--profile turbo-full-stack``), never over an
operator's export or a disabled lane, and never raises.
"""

from __future__ import annotations

from ..full_stack_env import stamp_import_time_defaults as _stamp_early

# Register the stacked auxiliary lane names (mtplx.qwen4_aux_lanes) with
# full_stack_env BEFORE the early stamp reads --disable-optimization /
# MTPLX_FABLE_DISABLE below, so an operator who turns one of those lanes off in
# the same command does not make the early stamp discard the whole disable list
# (parse_disable_lanes would otherwise reject the unknown name). The import is
# MLX-free and only populates the extra-lane table.
from .. import qwen4_aux_lanes as _qwen4_aux_lanes  # noqa: F401

__all__ = ["openai"]

#: What the early stamp actually put in place, for the receipt the server
#: prints once it has a stdout worth printing to. Empty on any other model
#: family, and empty when the operator had already exported all eleven (or
#: disabled their lanes).
EARLY_STAMPED_ENV: dict[str, str] = _stamp_early()
