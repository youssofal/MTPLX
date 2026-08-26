# RAMP evidence trail

This directory is the full, unedited investigation behind the RAMP feature in
`mtplx/context_copy.py` — including the parts that were wrong and got
corrected, and the parts that were proposed and killed. Nothing here was
cleaned up after the fact; that is deliberate (see the process note at the
bottom).

**Start here if you just want the result:** the main `README.md`'s "RAMP"
section has the summary and the env vars. This directory is for verifying the
numbers yourself.

## Reading order

1. **`SC-P4DA-ramp-serving-path-proposer.md`** — the original design + POC
   proposal: what problem RAMP solves, what was tried, what was measured.
2. **`POC-FINDINGS.md`** — the short-context (~800 token) proof-of-concept
   results. Two of four originally-proposed mechanisms were measured and
   killed here (indexing generated output: -0.7%; consensus ranking: +0.0%;
   agreement-based sizing: -52.4%) — the feature that shipped is the
   narrower survivor.
3. **`VERDICT-SC-P4DA.md`** — an independent adversarial review of the above.
   Confirms the core no-diff seam finding; catches several real overstatements
   (precision claims, an unquoted methodological objection, an unsatisfiable
   acceptance criterion) before the card was allowed to ship.
4. **`006-ramp-ema-guard-hazard-and-block-length.md`** — records a known,
   measured hazard (the engine's safety-suspend guard fires more often as
   block length grows) rather than silently shipping around it.
5. **`longcontext-investigation/`** — what happens when the short-context
   result is checked against the actual target workload (128K-256K context).
   Three consecutive modelling rounds (`007`, `SC-P4DB`, `SC-P4DC`) each
   found and corrected a real error in the previous round's math — including
   a proposed production fix (`SC-P4DB`'s "proposal floor") that was
   ultimately ruled **INVALID_CARD** because it would have applied to zero
   shipped configurations. Read `VERDICT-SC-P4DC.md`'s closing recommendation
   if you want the reason the project stopped modelling and went to
   measurement instead.
6. **`008-ramp-live-128k-ground-truth.md`** — the live measurement that
   actually settled it: RAMP's advantage does not collapse at 128K, though it
   mildly shrinks rather than growing as the models predicted (+51% short
   context vs +46-71% at 128K, depending on run cleanliness — the modelling
   chain's directional confidence was not fully earned).

## Process note

Every rejected model, killed mechanism, and adversarial verdict is committed
here on purpose, including the ones that make earlier claims in this same
directory look wrong. A reader who only sees the polished final numbers
cannot judge how much scrutiny they survived. If you are deciding whether to
trust this feature's defaults, the disagreements are the evidence — not the
summary.
