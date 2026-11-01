# Research poster layout

A0 portrait (841 x 1189 mm), three columns. Readable from two metres: body text 28–32 pt, headings 60 pt, title 100 pt.
Every number below comes from `reports/` (run on the real UCI data); the figures are already in `docs/figures/`.

```
+----------------------------------------------------------------------------------+
|  WHEN CAN FAIRNESS MITIGATION WORK? A pre-flight check for student-success models |
|  MD SADRUL ANAM · Gannan Normal University · github.com/<you>/fairpath  [QR code] |
+--------------------------+---------------------------+---------------------------+
| 1. WHY THIS MATTERS      | 3. MITIGATION, COMPARED   | 5. THE PRE-FLIGHT CHECK   |
| Early-warning models     |  [figure: Pareto frontier,| Three questions before    |
| decide who gets an       |   dropout, gender]        | you mitigate:             |
| advisor's call. Overall  |                           |  1 Precise? CI half-width |
| accuracy hides who is    | Every family cut the      |    <= 0.05                |
| missed.                  | equalized-odds gap on     |  2 Real? permutation      |
|                          | dropout (Holm p = 0.016,  |    p < 0.05               |
| Question: under what     | all 8 vs baseline).       |  3 Worth it? CI lower     |
| data conditions can      |                           |    bound >= 0.05          |
| mitigation measurably    | On student-mat nothing    |                           |
| work?                    | non-degenerate helped;    | Dropout x gender: MITIGATE|
|                          | "zero gap" = flag all /   | Student-mat x sex:        |
| 2. DATA & PROTOCOL       | flag none.                |   INSUFFICIENT DATA       |
|  Dropout n=3,630 (UCI)   |                           |                           |
|  Student-mat n=395 (UCI) | 4. HELD-OUT RESULT        | 6. TAKEAWAYS              |
|  Leaking grade columns   |  [figure: group bars,     |  * Measure before you     |
|  removed: AUC 0.954 ->   |   baseline vs deployed]   |    mitigate.              |
|  0.845 (5x2 CV).         |                           |  * Blinding alone leaves  |
|  5 tuned models, 9       |  Equalized odds           |    most of the gap        |
|  mitigation conditions,  |    0.328 -> 0.150         |    (proxies).             |
|  5x2 CV, bootstrap,      |  Equal opportunity        |  * Reject "fair" models   |
|  permutation, Holm,      |    0.184 -> 0.036         |    that flag everyone.    |
|  corrected t-test.       |  ROC-AUC 0.864 -> 0.861   |                           |
|                          |  (deployed model never    | [dashboard screenshot]    |
|                          |   sees gender)            |                           |
+--------------------------+---------------------------+---------------------------+
| References (4–5) · Data: Realinho et al. 2022; Cortez & Silva 2008 (CC BY 4.0)   |
+----------------------------------------------------------------------------------+
```

## Design notes

* **One idea per column.** Left column: problem and method. Middle: evidence. Right: the contribution and what to do with it.
  A reader who only reads the right column should still leave with the main message.
* **Colour.** Use the dashboard's palette: blue `#2a78d6` for the deployed model, orange `#eb6834` for the baseline, grey
  for everything else. Never encode a group only by colour; label bars directly.
* **Figures to use.** `docs/figures/pareto_dropout.png` (middle top), `docs/figures/groups_dropout.png` (middle bottom),
  `docs/screenshots/fairness.png` (right bottom, cropped). Re-export at 300 dpi for print with
  `fairpath report` after raising `dpi` in `src/fairpath/pipeline/figures.py`.
* **The verdict box** in column 3 is the visual anchor: a two-row table, green "Mitigate" and amber "Insufficient data", with
  the word, not just the colour.
* **QR code** to the live demo in the top-right corner. People scan it during the session; it is the best way to show the
  dashboard on a printed poster.
* **Honesty line** in small print under the title: "Results on public UCI data. The live demo serves models trained on the same
  data; examples shown are synthetic."
