# Plan: Uzi Lane-Dominance Mini Paper

Slug: `uzi-lane-paper`

## Deliverables

1. Mini paper in NeurIPS 2026 LaTeX format, 4--8 content pages, plus references and appendix.
2. Poster source and exported PDF/PPT-ready artifact for the June 30, 2026 poster session.
3. Code package manifest with the exact repo state, selected commit history, logs, generated figures, and synthetic probe scripts/results.

## Core Story

We will frame the project as training a lane-dominant 1v1 Honor of Kings agent, nicknamed "Uzi", whose final behavior prioritizes early lane control, precise target selection, safe pressure, and tower conversion. The name intentionally references the League of Legends professional player Uzi: a mechanically aggressive laner who wins by pressure, spacing, trading, and conversion rather than passive scaling. The story is not "we solved every mechanic"; it is "we iteratively turned a noisy multi-discrete PPO policy into a strong lane bully by aligning entity features, target semantics, reward shaping, action legality, and curriculum."

Key honest claims:

- The final local `ppo-5-1` artifact currently only contains an empty error-log pull, not a full metrics summary. I will attempt to recover/pull metrics if possible; otherwise I will report that final run logs show no pulled warnings/errors and use earlier PPO summaries as the quantitative training evidence.
- The recall experiments are a negative result. The code history shows recall reward shaping, legal-mask opening, forced exploration, hold assistance, and recall-channel state features, followed by disabling recall in the final config. The paper will present this as a case study in sparse multi-step behavior that did not survive policy learning.
- The Arli/Gongsun Li win-rate failure will be explained through concrete code/probe evidence: Arli has hero-specific mark/buff dynamics (`19900`) and mobility/state semantics that require stable state encoding; a single shared lane-bully objective and limited hero-specific rollout coverage can underfit her. I will avoid claiming an unverified root cause as fact; the paper can say our diagnostics suggest this failure mode.

## Evidence to Collect

### Repository and Git History

- Summarize commits from the PPO transition:
  - `436d5ad`: PPO uses DIY-style features.
  - `7b86846`: hybrid PPO encoder.
  - `f5ce14f`: PPO owns migrated features/rewards.
  - `4c07829`: training observability/action masks.
  - `ba56882`: multi-hero curriculum.
  - `49876d4`, `22cf711`: action shaping and metric pulls.
  - `a824774` through `840fd3e`: recall shaping, exploration, legality, staged training, monitoring, channel state.
- Use `git show` and selective file snapshots to tie each design stage to a measurable failure mode or fix.

### Real Logs

- Parse `logs/ppo-*/summary.json` and `metrics.csv`.
- Main comparison table: reward, win rate, kill/death, tower HP, money per frame, action Button9 count/rate where available.
- Curves:
  - win rate over run index/time;
  - algorithm reward over run index/time;
  - enemy tower HP and self tower HP;
  - Button9/recall action count in recall-enabled experiments;
  - loss/entropy/grad norm where available.
- Include a log availability note for `logs/ppo-5-1-errors.jsonl.summary.json`: 0 warning/error entries pulled from task `ppo-5-1`.

### Submitted Ranking Result

- Treat the user-provided final leaderboard result as the headline external evaluation:
  - Team: `Uzi`
  - Rank: 3
  - A wins / B wins: `131 / 67`
  - Total games: `198`
  - Aggregate win rate: `131 / 198 = 66.2%`
- Use this as the paper's main final-performance number, while keeping training-log plots as process evidence and synthetic probes as diagnostic evidence.
- Explain the name in the introduction: our design goal was an "Uzi-style" policy, i.e., a pressure-first laning agent that creates kill/tower threats through aggressive but legal target selection.

### Real Diagnostic Frames

- Use `diag_feature_probes/FEATURE_PROBE_SUMMARY.md` as primary evidence for:
  - Soldier1--4 target ordering: nearest four visible enemy soldiers, then ascending `runtime_id`.
  - Arli mark `19900` observed on minions with layers 0--3.
  - bullet observations and ability bits as useful dynamic state.
  - semantic attack-target relations.
- Convert the probe findings into an appendix table and one main-paper figure/table.

### Historical Synthetic Frame Probes

Create `outputs/uzi-lane-paper/probes/` with a small script that can run the same synthetic frame scenarios against selected historical commits using temporary git worktrees or `git show` snapshots.

Probe scenarios:

1. Target semantics probe:
   - Frame with three enemy minions whose distance order differs from runtime-id order.
   - Measure enemy-minion token ordering and target slot mapping in older vs newer code.
   - Expected story: old distance-only assumptions can make Button3/SoldierN point to the wrong entity; newer targeting aligns feature/reward/pointer semantics.

2. Normal attack legality probe:
   - Legal mask where Button3 has only target None/Self.
   - Measure whether the code suppresses invalid normal attack targets and masks Button3.
   - Expected story: later action-mask logic prevents PPO from wasting probability mass on impossible/meaningless attacks.

3. Recall probe:
   - Low-HP, enemy-far, safe-under-own-tower state.
   - Compare reward/action-mask behavior across `a824774`, `19fd508`, `3463df5`, `357627a`, `840fd3e`, and HEAD.
   - Expected story: code could create recall incentives and even force exploration, but the final policy/config disables recall because sparse channel completion competed with simpler retreat/heal/tower-pressure behavior.

4. Arli state probe:
   - Frame carrying Arli `19900` mark layers on minions and dynamic ability bits.
   - Measure whether the feature vector exposes these signals in relevant commits.
   - Expected story: Arli-specific mechanics are observable but sparse and hero-specific; the shared objective likely overfits easier marksman behavior, explaining a zero-win-rate failure case.

All synthetic outputs will be labeled as "synthetic diagnostic probes", not platform evaluation metrics.

## Paper Outline

Title: `Training Uzi: PPO for Lane-Dominant 1v1 Honor of Kings Agents`

Abstract:

- State the problem: 1v1 MOBA control with sparse terminal objective, multi-discrete action space, entity targets, and heterogeneous heroes.
- State method: structured entity features, button-conditioned target pointer, reward shaping, action-mask repair, dynamic opponent/hero curriculum, diagnostics.
- State result: strong lane pressure in PPO logs, with honest negative cases for recall and Arli.

1. Introduction

- Motivation: training a mechanically aggressive lane agent.
- Contributions:
  - entity-aware PPO architecture;
  - target/action legality alignment;
  - reward/curriculum/observability stack;
  - failure analysis for recall and Arli.

2. Environment and Task

- 1v1 Honor of Kings setup, heroes 112/133/199.
- Multi-discrete actions: button, movement/skill heads, target head.
- Win/tower/kill/economy objectives.

3. Method

- Observation design:
  - tokenized heroes, towers, minions, monsters, bullets, cakes, plus global features;
  - existence masks and fog-aware memory;
  - hero-specific state encoding.
- Policy architecture:
  - raw MLP residual plus token encoder;
  - AdaLN-conditioned token blocks;
  - LSTM residual;
  - button-conditioned target pointer.
- PPO objective:
  - clipped policy loss, value loss, entropy regularization.
- Reward shaping:
  - tower HP, HP trade, lane progress/presence, last hit, danger, death, tower attack, distance/no-op penalties.
- Curriculum and observability:
  - multi-hero round-robin;
  - dynamic common-AI/self-play mixture;
  - monitor panels for action quality, target distribution, entropy, gradients, reward components.

4. Experiments

- Data sources and limitations.
- Training-stage comparison table from logs.
- Curves and summary statistics.
- Ablation-style historical comparison from commits and synthetic probes.

5. Case Studies

- Case 1: Target semantics and lane aggression.
- Case 2: Why recall did not train.
- Case 3: Why Arli/Gongsun Li failed despite the shared model.
- Each case study includes a synthetic frame, code-history evidence, and a table/figure.

6. Limitations

- No full final `ppo-5-1` metrics currently available locally unless recovered.
- Synthetic probes diagnose code behavior, not full environment win rate.
- Multi-hero policy may require hero-conditioned objectives or per-hero fine-tuning.

7. Conclusion

- Summarize the engineering lesson: the winning behavior came from aligning target semantics, action legality, reward scale, and diagnostics more than from a single PPO trick.

Appendix:

- More log tables.
- Full synthetic probe outputs.
- Code package manifest.
- NeurIPS checklist if needed by template.

## Figures and Tables

Main paper target figures/tables:

1. Architecture diagram: observation tokens -> hybrid encoder -> LSTM -> multi-head PPO -> button-conditioned target pointer.
2. Training progression table across PPO runs.
3. Metric curves: win rate, reward, tower HP pressure, kill/death.
4. Reward component table with design intent and failure protection.
5. Historical synthetic probe table across commits.
6. Case study figure/table for recall.
7. Case study figure/table for Arli.

Poster:

- One central pipeline/architecture graphic.
- One training-progress panel.
- Three compact case-study boxes: target semantics, recall failure, Arli failure.

## Verification Log

- Verify all local metrics by parsing JSON/CSV, not manual copying.
- Verify synthetic probes by saving exact commit hash, scenario JSON, script version, stdout JSON, and generated summary table.
- Verify LaTeX compiles under the provided NeurIPS 2026 template.
- Verify poster renders to PDF or high-resolution images.
- Verify code package includes only necessary code/log/report artifacts and does not overwrite user changes.

## File Plan

- `outputs/uzi-lane-paper/paper/`: NeurIPS LaTeX, figures, bibliography, compiled PDF.
- `outputs/uzi-lane-paper/poster/`: poster source and export.
- `outputs/uzi-lane-paper/probes/`: synthetic frame scenarios, runner, results.
- `outputs/uzi-lane-paper/code-package/`: manifest and zipped package.
- `papers/uzi-lane-paper.md`: optional Markdown draft generated before final LaTeX.
