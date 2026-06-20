import importlib
import sys
import types
import unittest


class _FakeMonitorConfigBuilder:
    def __init__(self):
        self.metrics = []
        self.exprs = {}

    def title(self, *_args, **_kwargs):
        return self

    def add_group(self, *_args, **_kwargs):
        return self

    def end_group(self):
        return self

    def add_panel(self, *_args, **_kwargs):
        return self

    def end_panel(self):
        return self

    def add_metric(self, metrics_name, *_args, **kwargs):
        self.metrics.append(metrics_name)
        if "expr" in kwargs:
            self.exprs[metrics_name] = kwargs["expr"]
        return self

    def build(self):
        return {"metrics": list(self.metrics), "exprs": dict(self.exprs)}


def _install_fake_monitor_builder():
    sys.modules.setdefault("kaiwudrl", types.ModuleType("kaiwudrl"))
    sys.modules.setdefault("kaiwudrl.common", types.ModuleType("kaiwudrl.common"))
    sys.modules.setdefault("kaiwudrl.common.monitor", types.ModuleType("kaiwudrl.common.monitor"))
    builder_module = types.ModuleType("kaiwudrl.common.monitor.monitor_config_builder")
    builder_module.MonitorConfigBuilder = _FakeMonitorConfigBuilder
    sys.modules["kaiwudrl.common.monitor.monitor_config_builder"] = builder_module


def _monitor_exprs():
    _install_fake_monitor_builder()
    module = importlib.import_module("agent_ppo.conf.monitor_builder")
    module = importlib.reload(module)
    return module.build_monitor()["exprs"]


class AgentPpoMonitorTests(unittest.TestCase):
    def test_recall_count_panels_keep_fractional_average_precision(self):
        exprs = _monitor_exprs()

        for metric in (
            "recall_need_cnt",
            "recall_start_cnt",
            "recall_hold_cnt",
            "recall_miss_cnt",
            "recall_interrupt_cnt",
            "recall_success_cnt",
            "recall_unneeded_cnt",
            "recall_explore_need_cnt",
            "recall_explore_legal_cnt",
            "recall_explore_forced_legal_cnt",
            "recall_explore_override_cnt",
            "recall_explore_hold_cnt",
        ):
            self.assertEqual(
                exprs[metric],
                "round(avg(%s{}), 0.001)" % metric,
            )


if __name__ == "__main__":
    unittest.main()
