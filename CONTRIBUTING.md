# Contributing

Thanks for taking a look. Issues and pull requests are welcome, especially:

- replications on other education datasets (the pre-flight check was derived from two Portuguese datasets, and I'd love to know where it breaks);
- new mitigation methods wired into `src/fairpath/mitigation.py` and the study in `pipeline/fairness_study.py`;
- accessibility or usability fixes in the dashboard.

## Getting set up

```bash
make setup          # Python deps, editable install, pre-commit hooks
make train-fast     # whole pipeline on the synthetic data, a few minutes
make test           # unit + API tests
```

For the real numbers: `make data && make train` (about 25 minutes on two CPU cores).

## Ground rules

1. **No raw student data in git.** The pre-commit hook blocks `data/raw/*.csv`. Use the synthetic data in tests.
2. **No hand-typed results.** Any number that appears in docs or the dashboard must come from `reports/*.json`, which the pipeline writes. If you change the pipeline, re-run it and commit the regenerated reports.
3. **Keep the test set sacred.** Anything that tunes, selects or calibrates must only touch the development split.
4. **Fairness changes need intervals.** A new metric or method should report bootstrap CIs or a significance test, not just a point estimate.
5. Style is enforced by `ruff` and `black` (`make format`). Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:` ...).

## Reporting a problem with the model's behaviour

If you think the tool produces a harmful or biased output, please open an issue with the label `fairness` and, if you can, the input that caused it (synthetic or anonymised only).
