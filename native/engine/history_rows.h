#pragma once

#include "mlx/array.h"

namespace mtplx_engine {

// The caller has reserved enough rows at this logical cache offset.
mlx::core::array write_history_rows(
    const mlx::core::array& history, const mlx::core::array& rows, const mlx::core::array& offset);

} // namespace mtplx_engine
