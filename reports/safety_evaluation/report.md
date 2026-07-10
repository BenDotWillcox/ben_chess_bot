# Stockfish safety-layer evaluation

Seeded held-out sample: **200/200 successful**.
Seed 42 fixes the sampled positions and policy draws; the 50 ms Stockfish searches and latency are not deterministic.

- Veto rate: **14/200 (7.00%)**.
- Median/p95 centipawn loss before: **1.00 / 249.60**; after: **0.00 / 171.05**.
- Centipawn loss avoided, median/p95: **0.00 / 2.90** (11 positive samples).
- Conditional loss avoided on positive samples, median/p95: **645.00 / 100218.00**.
- Non-mate conditional loss avoided, median/p95: **77.00 / 605.00**.
- Non-mate mean loss before/after: **38.65 / 31.87**; mate-scale losses: **5 / 0**.
- Added safety latency, median/p95: **310.53 / 315.33 ms**.
- Draw/shuffle risks avoided: **8/12**.
- Held-out Ben move match before/after: **52.00% / 52.50%**.
- Original personalized choice preserved: **93.00%**.
- Full held-out history replay gate: **3638/3638 positions reconstructed (0 mismatches)**.

## Caveats

- Safety results apply only to the recorded seeded held-out subset and engine time budget.
- Seed 42 fixes subset selection and policy sampling, not finite-time Stockfish output.
- Centipawn values, veto decisions, and latency are timing- and hardware-dependent and can vary across reruns.
- Ben-likeness is held-out move agreement, not a causal style or win-rate estimate.
- Draw/shuffle rates can be sparse; the report retains the raw numerator and denominator.
- Mate scores use python-chess's 100,000-centipawn sentinel; mate-scale and non-mate distributions are reported separately.
