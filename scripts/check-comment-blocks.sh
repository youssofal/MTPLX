#!/usr/bin/env bash
# Fail on comment blocks >$MAX consecutive lines or >$WIDTH chars per line.
# Override: CHECK_COMMENT_BLOCKS_MAX, CHECK_COMMENT_BLOCKS_WIDTH.
set -euo pipefail

MAX="${CHECK_COMMENT_BLOCKS_MAX:-3}"
WIDTH="${CHECK_COMMENT_BLOCKS_WIDTH:-88}"
status=0

check_file() {
    local f="$1"
    awk -v file="$f" -v max="$MAX" -v width="$WIDTH" '
        {
            line = $0; sub(/^[ \t]+/, "", line)
            is_comment = (line ~ /^#/) || (line ~ /^\/\//)
        }
        is_comment {
            block++
            if (block > max && !reported) {
                printf "%s:%d: comment block exceeds %d consecutive lines\n", file, NR - block + 1, max > "/dev/stderr"
                reported = 1
            }
            if (length(line) > width && !wide_reported) {
                printf "%s:%d: comment line exceeds %d chars (%d)\n", file, NR, width, length(line) > "/dev/stderr"
                wide_reported = 1
            }
            next
        }
        { block = 0 }
        END { if (reported || wide_reported) exit 1 }
    ' "$f" || return 1
    return 0
}

if [ $# -eq 0 ]; then
    while IFS= read -r f; do
        [ -f "$f" ] || continue
        check_file "$f" || status=1
    done
else
    for f in "$@"; do
        [ -f "$f" ] || continue
        check_file "$f" || status=1
    done
fi

exit $status
