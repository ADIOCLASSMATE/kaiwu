# Uzi Code Package Manifest

This package supports the mini-paper and poster submission for Team Uzi.

## Included code

- `agent_ppo/`: final PPO agent implementation used for the submitted model family.
- `agent_diy/`: earlier feature and reward implementation used during migration and diagnosis.
- `conf/`: application and algorithm configuration.
- `script/`: training data pullers, environment diagnostics, remote utilities, and probe scripts.
- `tests/`: focused regression tests for feature semantics, action masks, reward shaping, monitoring, and workflow behavior.
- `train_test.py`, `kaiwu.json`, `README.md`, `Diagnosis.md`: top-level run and diagnosis files.

## Included evidence

- `logs/ppo-5-1/`: final training metrics and pulled error logs. `errors.jsonl` is empty; the summary reports zero warning/error entries.
- `diag_feature_probes/`: real environment diagnostic frames and summary files.
- `outputs/uzi-lane-paper/data/`: parsed CSV/JSON analysis tables.
- `outputs/uzi-lane-paper/figures/`: generated vector figures.
- `outputs/uzi-lane-paper/probes/results/`: synthetic historical probe outputs.
- `outputs/uzi-lane-paper/paper/`: NeurIPS paper source and compiled PDF.
- `outputs/uzi-lane-paper/poster/`: poster source, images, and compiled PDF.

## Important notes

- The final leaderboard result used in the paper is: rank 3, Team Uzi, 131/67 over 198 games, 66.2% aggregate win rate.
- Synthetic probes are deterministic code diagnostics, not game win-rate evaluations.
- The working tree contained PPO code changes before the paper-generation work began; those changes are preserved in the package and are not reverted.
