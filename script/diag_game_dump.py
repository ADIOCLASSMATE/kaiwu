#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
完整对局诊断数据导出脚本。

在 Kaiwu WebIDE 终端执行：
  KAIWU_DUMP_SAMPLE_GAP=200 python3 script/diag_game_dump.py

输出目录：diag_dumps/<timestamp>/
每个采样帧一个 JSON 文件 + 一份摘要 SUMMARY.json。

KAIWU_DUMP_SAMPLE_GAP 控制采样间隔（帧），默认 500。
设置 KAIWU_DUMP_SAMPLE_GAP=99999 只会导出 start/mid/end 三个截面。
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

from kaiwudrl.common.utils.train_test_utils import run_train_test

# ---- 配置 ----
SAMPLE_GAP = int(os.environ.get("KAIWU_DUMP_SAMPLE_GAP", "500"))
OUTPUT_DIR = Path(os.environ.get("KAIWU_DUMP_DIR", "diag_dumps")) / datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)
PINNED_FRAMES = {0, 1, 2000, 5000, 10000, 15000}


# ---- 序列化 ----
class DiagEncoder(json.JSONEncoder):
    def default(self, o):
        try:
            import numpy as np

            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                if math.isnan(o) or math.isinf(o):
                    return None
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
        except ImportError:
            pass
        return super().default(o)


def _safe_dict(obj):
    try:
        import numpy as np
    except ImportError:
        np = None

    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, str):
        return obj
    if np is not None:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.ndarray):
            return _safe_dict(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _safe_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_dict(v) for v in obj]
    return str(obj)


# ---- 内部函数 ----
def _save_snapshot(collected, frame_no, observation):
    """保存当前帧的 raw observation 快照。去重：同一帧号只存一次。"""
    if frame_no in collected:
        return

    snap = {}
    for agent_id in observation:
        obs = observation[agent_id]
        snap[str(agent_id)] = {
            "camp": _safe_dict(obs.get("camp")),
            "player_camp": _safe_dict(obs.get("player_camp")),
            "player_id": _safe_dict(obs.get("player_id")),
            "frame_state": _safe_dict(obs.get("frame_state")),
            "legal_action": _safe_dict(obs.get("legal_action")),
            "sub_action_mask": _safe_dict(obs.get("sub_action_mask")),
            "reward": _safe_dict(obs.get("reward")),
            "score": _safe_dict(obs.get("score")),
        }

    collected[frame_no] = snap
    path = OUTPUT_DIR / f"frame_{frame_no:05d}.json"
    path.write_text(
        json.dumps(snap, ensure_ascii=False, indent=2, cls=DiagEncoder),
        encoding="utf-8",
    )


def _write_summary(collected):
    """写出 SUMMARY.json。"""
    summary = {
        "total_snapshots": len(collected),
        "frames": sorted(collected.keys()),
        "game_length": max(collected.keys()) if collected else 0,
        "per_frame_summary": {},
    }

    for fno in sorted(collected):
        snap = collected[fno]
        s = {}
        for aid in snap:
            fs = snap[aid].get("frame_state", {}) or {}
            heroes = fs.get("hero_states", []) or []
            npcs = fs.get("npc_states", []) or []
            bullets = fs.get("bullets", []) or []
            cakes = fs.get("cakes", []) or []

            # NPC 类型分布
            npc_types = {}
            for n in npcs:
                at = n.get("actor_type")
                st = n.get("sub_type")
                camp = n.get("camp")
                alive = n.get("hp", 0) > 0
                key = f"actor={at},sub={st}"
                if key not in npc_types:
                    npc_types[key] = {"total": 0, "alive": 0, "camps": {}}
                npc_types[key]["total"] += 1
                if alive:
                    npc_types[key]["alive"] += 1
                if camp is not None:
                    npc_types[key]["camps"][str(camp)] = (
                        npc_types[key]["camps"].get(str(camp), 0) + 1
                    )

            hero_summary = []
            for h in heroes:
                hero_summary.append(
                    {
                        "config_id": h.get("config_id"),
                        "camp": h.get("camp"),
                        "hp": h.get("hp"),
                        "max_hp": h.get("max_hp"),
                        "level": h.get("level"),
                        "money": h.get("money"),
                        "alive": h.get("hp", 0) > 0,
                        "location": h.get("location"),
                        "actor_type": h.get("actor_type"),
                        "sub_type": h.get("sub_type"),
                    }
                )

            s[aid] = {
                "frame_no": fs.get("frame_no"),
                "hero_count": len(heroes),
                "npc_count": len(npcs),
                "bullet_count": len(bullets),
                "cake_count": len(cakes),
                "heroes": hero_summary,
                "npc_types": npc_types,
            }

        summary["per_frame_summary"][str(fno)] = s

    path = OUTPUT_DIR / "SUMMARY.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, cls=DiagEncoder),
        encoding="utf-8",
    )
    print(f"[diag_dump] Summary written to {path}")


# ---- 主流程 ----
def main():
    import agent_diy.workflow.train_workflow as tw

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    collected = {}  # frame_no -> raw observation snapshot

    _orig_workflow = tw.workflow

    def dump_workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
        env = envs[0]
        env_conf_manager = tw.EnvConfManager(
            config_path="agent_diy/conf/train_env_conf.toml",
            logger=logger,
        )
        lineup_iterator = tw.lineup_iterator_roundrobin_camp_heroes([112, 133, 199])

        runner = tw.EpisodeRunner(
            env=env,
            agents=agents,
            logger=logger,
            monitor=monitor,
            env_conf_manager=env_conf_manager,
            lineup_iterator=lineup_iterator,
        )

        # ---- 对局初始化（完全复刻原始 workflow） ----
        lineup = next(runner.lineup_iterator)
        usr_conf, is_eval, monitor_side = runner.env_conf_manager.update_config(lineup)
        runner._call_init_config(usr_conf)

        env_obs = env.reset(usr_conf=usr_conf)
        observation = env_obs["observation"]
        runner.reset_agents(observation)

        frame_collector = tw.FrameCollector(runner.agent_num)

        is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
        logger.info(f"[diag_dump] Episode start, sampling every {SAMPLE_GAP} frames, output={OUTPUT_DIR}")

        # ---- 初始 reward（和原始 workflow 一致） ----
        for i, (do_sample, agent) in enumerate(zip(runner.do_samples, runner.agents)):
            if do_sample:
                reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                observation[str(i)]["reward"] = reward

        # 初始帧快照
        _save_snapshot(collected, 0, observation)

        while True:
            actions = [tw.NONE_ACTION] * runner.agent_num
            for index, (do_predict, do_sample, agent) in enumerate(
                zip(runner.do_predicts, runner.do_samples, runner.agents)
            ):
                if do_predict:
                    if not is_eval:
                        actions[index] = agent.predict(observation[str(index)])
                    else:
                        actions[index] = agent.exploit(observation[str(index)])

                    if not is_eval and do_sample:
                        frame = tw.build_frame(agent, observation[str(index)])
                        frame_collector.save_frame(frame, agent_id=index)

            env_reward, env_obs = env.step(actions)
            frame_no = env_obs["frame_no"]
            observation = env_obs["observation"]
            terminated = env_obs["terminated"]
            truncated = env_obs["truncated"]

            # ---- reward（和原始 workflow 一致） ----
            for i, (do_sample, agent) in enumerate(zip(runner.do_samples, runner.agents)):
                if do_sample:
                    reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                    observation[str(i)]["reward"] = reward

            # ---- 采样当前帧 ----
            if (frame_no % SAMPLE_GAP == 0) or (frame_no in PINNED_FRAMES):
                _save_snapshot(collected, frame_no, observation)

            is_gameover = terminated or truncated or (is_train_test and frame_no >= 1000)
            if is_gameover:
                _save_snapshot(collected, frame_no, observation)
                logger.info(
                    f"[diag_dump] Episode ended at fno={frame_no}, "
                    f"terminated={terminated}, truncated={truncated}, "
                    f"saved {len(collected)} snapshots"
                )
                break

        _write_summary(collected)
        logger.info(f"[diag_dump] Done. Output: {OUTPUT_DIR}")
        return  # 不 yield，结束整个脚本

    tw.workflow = dump_workflow

    run_train_test(
        algorithm_name="diy",
        algorithm_name_list=["ppo", "diy"],
        env_vars={
            "replay_buffer_capacity": "128",
            "preload_ratio": "1.0",
            "reverb_remover": "reverb.selectors.Fifo",
            "reverb_sampler": "reverb.selectors.Uniform",
            "reverb_rate_limiter": "MinSize",
            "reverb_samples_per_insert": "8",
            "reverb_error_buffer": "8",
            "train_batch_size": "32",
            "dump_model_freq": "1000",
            "model_file_sync_per_minutes": "1",
            "modelpool_max_save_model_count": "1",
            "preload_model": "False",
            "preload_model_dir": "{agent_name}/ckpt",
            "preload_model_id": "1000",
        },
    )


if __name__ == "__main__":
    main()
