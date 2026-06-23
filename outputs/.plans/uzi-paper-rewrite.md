Uzi paper rewrite evidence plan

Scope:
- Update `outputs/uzi-lane-paper/paper/main.tex` in place.
- Keep the paper grounded in code, generated poster figures, platform logs, and git history.
- Compile and render-check the final PDF.

Evidence used:
- Feature/code: `agent_ppo/conf/conf.py`, `agent_ppo/feature/feature_process/builder.py`, `agent_ppo/feature/action_mask.py`, `agent_ppo/feature/targeting.py`.
- Model/code: `agent_ppo/model/model.py`, `agent_ppo/agent.py`.
- Reward/training/code: `agent_ppo/feature/reward_process.py`, `agent_ppo/workflow/train_workflow.py`, `agent_ppo/conf/monitor_builder.py`.
- Tests: `tests/test_agent_ppo_feature_swap.py`.
- Outputs: `outputs/uzi-lane-paper/data/analysis_summary.json`, `outputs/uzi-lane-paper/data/historical_probe_summary.csv`.
- Git commits: feature migration, hybrid encoder, observability/action masks, recall shaping/exploration/state, poster asset exports.

Main edits:
- Fix figure references to use the new poster-selected high-resolution feature and policy architecture assets.
- Remove the misleading total reward plot from the main results.
- Add explicit feature layout and model architecture details from code.
- Explain why reward could appear to decline while behavior remained stable.
- Tie historical probe and commit evidence to target semantics, recall failure, and Arli failure.
