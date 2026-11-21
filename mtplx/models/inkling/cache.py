"""Window state saved at a conversation boundary, including its physical write cursor."""

from mlx_lm.models.cache import RotatingKVCache


class WindowCache(RotatingKVCache):
    # The same number of rows can describe different absolute positions and ring orders.
    requires_restore_meta = True

    def is_trimmable(self):
        # A short prefix is still present now, but may be overwritten by the time it is restored.
        return False

    @RotatingKVCache.state.getter
    def state(self):
        if self.keys is not None and self.keys.shape[2] > self.max_size:
            # Multi-token attention used these extra rows for its earliest queries. Only
            # the last window can be read by the first query after this checkpoint.
            return self.keys[:, :, -self.max_size :], self.values[
                :, :, -self.max_size :
            ]
        return super().state

    @RotatingKVCache.meta_state.getter
    def meta_state(self):
        return tuple(
            map(
                str,
                (self.keep, self.max_size, self.offset, min(self._idx, self.max_size)),
            )
        )
