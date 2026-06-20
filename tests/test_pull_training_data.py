from script.pull_training_data import build_summary, default_monitor_paths


def test_build_summary_keeps_future_metrics_in_misc_group():
    metrics_data = {
        "reward": {"has_data": "yes", "last": 12.5},
        "future_metric_v2": {"has_data": "yes", "last": 0.75},
        "future_zero_metric": {"has_data": "all_zero", "last": 0.0},
        "future_no_data_metric": {"has_data": "no", "last": None},
    }
    summary = build_summary(
        {"id": 1, "name": "task", "status": "done"},
        metrics_data,
        set(metrics_data),
        "2026-06-20T00:00:00Z",
        "2026-06-20T01:00:00Z",
    )

    assert summary["latest"]["algorithm"]["reward"] == 12.5
    assert "reward" not in summary["latest"]["misc_metrics"]
    assert summary["latest"]["misc_metrics"]["future_metric_v2"] == 0.75
    assert summary["latest"]["misc_metrics"]["future_zero_metric"] == 0.0
    assert "future_no_data_metric" not in summary["latest"]["misc_metrics"]


def test_default_monitor_paths_only_use_ppo_monitor():
    paths = default_monitor_paths()

    assert len(paths) == 1
    assert paths[0].as_posix().endswith("agent_ppo/conf/monitor_builder.py")
    assert "agent_diy" not in paths[0].as_posix()
