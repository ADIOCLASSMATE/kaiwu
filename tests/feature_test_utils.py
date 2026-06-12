import copy
import json
from pathlib import Path

from agent_diy.conf.conf import FeatureConfig as FC


ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "diag_feature_probes"


def load_obs(relative_path):
    data = json.loads((PROBES / relative_path).read_text(encoding="utf-8"))
    return next(iter(data["observation"].values()))


def iter_probe_observations():
    for path in sorted(PROBES.glob("episode_*/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        yield path, next(iter(data["observation"].values()))


def copied_frame(observation):
    return copy.deepcopy(observation["frame_state"])


def token_slice(feature, type_key, index=0):
    token_range = FC.TOKEN_SLICES[type_key][index]
    return feature[token_range]


def token_field(token, field_name, field_slices):
    return token[field_slices[field_name]]


def hero_field(token, field_name):
    return token_field(token, field_name, FC.HERO_FIELD_SLICES)


def minion_field(token, field_name):
    return token_field(token, field_name, FC.MINION_FIELD_SLICES)


def cake_field(token, field_name):
    return token_field(token, field_name, FC.CAKE_FIELD_SLICES)


def main_hero(frame_state, camp):
    return next(hero for hero in frame_state["hero_states"] if hero["camp"] == camp)


def enemy_hero(frame_state, camp):
    return next(hero for hero in frame_state["hero_states"] if hero["camp"] != camp)


def set_slot_config(hero, slot_type, config_id):
    for slot in (hero.get("skill_state") or {}).get("slot_states") or []:
        if slot.get("slot_type") == slot_type:
            slot["configId"] = config_id
            return
    raise AssertionError(f"missing slot_type={slot_type}")
