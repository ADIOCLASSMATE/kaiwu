# Agent DIY Feature Probe Summary

Date: 2026-06-09

## Artifacts

- `SUMMARY.json`: aggregate event counts and probe statistics.
- `events.jsonl`: field changes, bullet tracks, and target probe events.
- `target_probes.json`: issued Soldier actions and resolved `real_cmd.actorID`.
- `episode_*/frame_*.json`: selected raw observations used to verify conclusions.
- `../script/diag_feature_probe.py`: collection script.

## Confirmed Findings

### 1. Soldier pointer ordering is wrong

`agent_diy` currently sorts enemy minion tokens by distance and maps them directly to
Soldier1-4. The environment target slots do not use that order.

Three probes resolved an actual actor ID:

| Probe | Slot | Distance order | Runtime ID order | Actual |
|---|---|---|---|---|
| 21 | Soldier1 | `[20, 22, 24]` | `[20, 22, 24]` | `20` |
| 25 | Soldier3 | `[101, 110, 106]` | `[101, 106, 110]` | `110` |
| 38 | Soldier3 | `[479, 486, 474]` | `[474, 479, 486]` | `486` |

All three match ascending `runtime_id`. Probes 25 and 38 explicitly contradict
distance order. Raw evidence:

- `episode_02/frame_01286.json` and `frame_01292.json`
- `episode_02/frame_01802.json` and `frame_01820.json`
- `episode_02/frame_05264.json` and `frame_05270.json`

Conservative implementation rule:

1. Filter visible, alive enemy soldiers.
2. Select the nearest four, matching the environment documentation.
3. Sort that selected set by ascending `runtime_id` for Soldier1-4.

The selection rule when more than four soldiers are visible was not directly
probed. Only the per-slot ordering is confirmed.

The same ordering helper must be shared by:

- FeatureBuilder enemy minion token construction.
- Pointer target slot mapping.
- Reward/out-of-range target resolution.

### 2. Bullet data is useful and observable

The run observed 6,860 bullet records:

- 1,944 from visible enemy heroes.
- Enemy hero bullets occurred for all three heroes.
- Observed hero bullet `slot_type` values covered 0-3.
- Repeated runtime IDs across frames allow velocity estimation.
- `skill_id` was always zero in this run and should not be relied on.

Raw examples:

- Luban: `episode_01/frame_01304.json`
- Di Renjie: `episode_02/frame_01214.json`, `frame_01220.json`
- Arli: `episode_03/frame_01874.json`, `frame_01880.json`

Useful features are source camp/type, source hero identity, slot type, relative
position, distance, and cross-frame velocity. Enemy hero bullets should have
priority over minion projectiles.

### 3. Abilities contain dynamic hero state

Observed hero ability indices:

- Index 1: documented as `NoMove`.
- Index 5: documented as `NoMoveRotate`.
- Indices 31 and 33: observed on heroes but absent from the current documentation.

Examples:

- Di Renjie index 31: `episode_02/frame_01214.json`, cleared in `frame_01220.json`.
- Arli index 33: `episode_03/frame_01874.json`, still set in `frame_01880.json`.

Do not invent names for indices 31 and 33. Encode stable raw bits or an
explicit documented/unknown grouping. The full array length observed was 46.

### 4. Buff marks carry real stack information

Arli mark `configId=19900` was observed with layers 0, 1, 2, and 3. In this run
the mark was commonly attached to minions, so hero-only encoding would miss it.

Raw examples:

- Layer 1: `episode_03/frame_00848.json`
- Layer 3: `episode_03/frame_00914.json`
- Layer 0: `episode_03/frame_01118.json`

All 14,004 observed `buff_skills.times` values were zero. Do not use `times` as
the primary signal. Prefer whitelisted buff presence and relevant mark layers.

### 5. Attack target relationships are useful

Observed nonzero relationships included:

- Hero -> hero, soldier, and tower.
- Tower -> soldier.
- Soldier -> hero, soldier, and tower.

At episode 2 frame 5258, enemy soldier runtime ID 479 had
`attack_target=7`, the controlled hero:

- `episode_02/frame_05258.json`

Encode semantic relations such as `targets_me`, `targets_enemy_hero`,
`targets_own_tower`, and `has_target`. Do not feed raw runtime IDs as numeric
features.

### 6. Revive time remains unverified

All three episodes ended without a hero death. Every captured `revive_time`
value was zero. The field exists in the protocol, but its scale and countdown
behavior still need a death-focused probe.

It is reasonable to add a clipped/soft-normalized revive timer, but document
the scale as provisional and keep the raw-field handling robust.

## Implementation Priority

1. Fix Soldier1-4 ordering and share the ordering helper everywhere.
2. Add hero ability state.
3. Add semantic `attack_target` relations.
4. Add bounded enemy hero bullet tokens with velocity memory.
5. Add relevant buff presence and mark layers, including minion marks.
6. Add provisional `revive_time`.

## Engineering Constraints

- Preserve camp mirroring and fog-of-war rules.
- Never update enemy memory from invisible ground-truth state.
- Keep token `exists` as the sole padding-mask signal.
- Bound all variable-length collections with deterministic selection/order.
- Update `FeatureConfig` dimensions and model-derived dimensions together.
- Feature dimension changes invalidate old checkpoints; training must restart.
- Add focused tests using the raw frames in this directory.
- Do not treat undocumented ability bits or arbitrary buff IDs as known semantics.

## Remaining Questions

- When more than four enemy soldiers are visible, does the environment first
  choose the nearest four and then sort by runtime ID? This is the best current
  inference, not directly proven.
- What do ability indices 31 and 33 mean?
- What is the unit and maximum range of `revive_time`?
- Which buff IDs are decision-critical beyond Arli mark `19900`?

