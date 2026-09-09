# Table 4 — Supplementary knock-outs on arm C at 16K (fastest of 3)

These knock-outs are supplementary. The plan was ten lanes. David cancelled
the run after three windows. These three windows finished before the cancel
and passed their gates. Each row removes one lane from arm C. The decode is
the fastest seed of that window. The delta is the fastest knock-out decode
minus arm C's fastest ABAB decode. A negative delta means the lane helps: its
removal makes decode slower. The single window gives a wide per-seed range,
so read the delta as an indication, not a paired result.

Arm C fastest ABAB decode is 82.84 tokens per second.

| lane | class | KO decode (range) | Δ vs C fastest | Δ% | gate rc |
| --- | --- | --- | ---: | ---: | ---: |
| hc_m4 | PR opt | — | — | — | 0 |
| qsa_sparse_decode | PR opt | — | — | — | 0 |
| mask_fuse | PR opt | — | — | — | 0 |

The gate on each window checked that the removed lane read off and the other
nine read on. hc_m4 read armed:false. qsa_sparse_decode dropped from the
install report. mask-fuse gave no engaged line in the server log.
