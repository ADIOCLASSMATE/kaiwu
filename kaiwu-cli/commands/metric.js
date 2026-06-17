import { api, contextFromArgs, apiPrefix, DOMAIN_ARGS, loadSession } from '../_session.js';

// metric_name -> PromQL expr 模板（2026-05-08 通过 group by (__name__) 实测拿到 37 个真名，修正所有 expr）
const METRIC_PROFILES = {
  // basic（actor / learner 累计计数器）
  predict_succ_cnt: 'sum(max_over_time(kaiwu_actor_predict_succ_cnt{}[1h]))',
  train_global_step: 'sum(kaiwu_train_global_step{})',
  load_model_succ_cnt: 'sum(kaiwu_actor_load_last_model_succ_cnt{})',
  sample_receive_cnt: 'sum(kaiwu_sample_receive_cnt{})',
  train_success_cnt: 'sum(kaiwu_train_success_cnt{})',
  episode_cnt: 'sum(max_over_time(kaiwu_episode_cnt{}[1h]))',
  sample_production_and_consumption_ratio: 'avg(kaiwu_sample_production_and_consumption_ratio{})',

  // algorithm（PPO loss / reward）
  reward: 'avg(kaiwu_reward{})',
  total_loss: 'avg(kaiwu_total_loss{})',
  value_loss: 'avg(kaiwu_value_loss{})',
  policy_loss: 'avg(kaiwu_policy_loss{})',
  entropy_loss: 'avg(kaiwu_entropy_loss{})',

  // env（self-play 监控指标）
  win_rate: 'avg(kaiwu_win_rate{})',
  self_tower_hp: 'avg(kaiwu_self_tower_hp{})',
  enemy_tower_hp: 'avg(kaiwu_enemy_tower_hp{})',
  frame: 'avg(kaiwu_frame{})',
  money_per_frame: 'avg(kaiwu_money_per_frame{})',
  kill: 'avg(kaiwu_kill{})',
  death: 'avg(kaiwu_death{})',
  hurt_by_hero: 'avg(kaiwu_hurt_by_hero{})',
  hurt_to_hero: 'avg(kaiwu_hurt_to_hero{})',

  // 性能 / 队列
  batch_train_cost_time_ms: 'avg(kaiwu_batch_train_cost_time_ms{})',
  real_train_cost_time_ms: 'avg(kaiwu_real_train_cost_time_ms{})',
  data_fetch_cost_time_ms: 'avg(kaiwu_data_fetch_cost_time_ms{})',
  aisrv_learner_proxy_queue_len: 'avg(kaiwu_aisrv_learner_proxy_queue_len{})',
  reverb_ready_size: 'avg(kaiwu_reverb_ready_size{})',
  max_sample_size: 'avg(kaiwu_max_sample_size{})',
  sample_consume_rate: 'avg(kaiwu_sample_consume_rate{})',
  sample_product_rate: 'avg(kaiwu_sample_product_rate{})',

  // 错误计数
  push_to_cos_err_cnt: 'sum(kaiwu_push_to_cos_err_cnt{})',
  push_to_cos_succ_cnt: 'sum(kaiwu_push_to_cos_succ_cnt{})',
  push_to_model_pool_err_cnt: 'sum(kaiwu_push_to_model_pool_err_cnt{})',
  push_to_model_pool_succ_cnt: 'sum(kaiwu_push_to_model_pool_succ_cnt{})',
  pull_from_model_pool_err_cnt: 'sum(kaiwu_pull_from_model_pool_err_cnt{})',
  pull_from_model_pool_succ_cnt: 'sum(kaiwu_pull_from_model_pool_succ_cnt{})',
  send_to_reverb_err_cnt: 'sum(kaiwu_send_to_reverb_err_cnt{})',
  send_to_reverb_succ_cnt: 'sum(kaiwu_send_to_reverb_succ_cnt{})',
};

const ALL_METRIC_NAMES = Object.keys(METRIC_PROFILES);

export default {
  name: 'metric',
  description: '拉训练任务的指标范围（GetTrainMetricRange）。默认查所有已知指标，输出最新值 / 是否有数据 / 数据点数',
  example: 'kaiwu metric --task-id 145229',
  domain: 'tencentarena.com',
  args: [
    { name: 'task-id', type: 'int', required: true, help: '训练任务 id' },
    ...DOMAIN_ARGS,
    { name: 'team-id', type: 'int', default: 0, help: 'competition team id；course 接口不需要' },
    { name: 'names', type: 'string', default: '', help: `指标名（逗号分隔），缺省查全部 37 个平台指标。自定义指标（如 feat_*, rwd_*, entropy_head_* 等）直接传名即可，自动加 kaiwu_ 前缀查询。可用平台指标: ${ALL_METRIC_NAMES.join(',')}` },
    { name: 'expr', type: 'string', default: '', help: '直接传 PromQL 表达式（与 --names 互斥）' },
    { name: 'step', type: 'int', default: 15, help: '采样步长（秒）' },
    { name: 'start', type: 'string', default: '', help: 'ISO8601 起始时间，缺省取 task start_time' },
    { name: 'end', type: 'string', default: '', help: 'ISO8601 结束时间，缺省取 task end_time（或当前）' },
    { name: 'raw', type: 'boolean', default: false, help: '输出原始数据点数组' },
    { name: 'dump-items', type: 'boolean', default: false, help: 'dump 原始 items 含 labels（探 metric 名时用）' },
  ],
  columns: ['name', 'has_data', 'points', 'min', 'max', 'last', 'expr'],
  run: async (kwargs) => {
    const session = await loadSession();
    const ctx = contextFromArgs(session, kwargs);
    const prefix = apiPrefix(ctx);
    const taskId = kwargs['task-id'];

    // 取 task start/end
    let startISO = kwargs.start;
    let endISO = kwargs.end;
    if (!startISO || !endISO) {
      const list = await api(`${prefix}/ListTrainTask`,
        { ...ctx, page: { current: 1, size: 50 } }, { session });
      const t = (list.train_task || []).find(x => x.id === taskId);
      if (!t) throw new Error(`task-id=${taskId} 不在最近 50 条任务里`);
      startISO = startISO || t.start_time;
      endISO = endISO || t.end_time || new Date().toISOString();
    }

    // 构造 queries
    let queries;
    if (kwargs.expr) {
      queries = [{ name: 'custom', expr: kwargs.expr, id: 'custom_0', step: String(kwargs.step) }];
    } else {
      const wanted = (kwargs.names || '').split(',').map(s => s.trim()).filter(Boolean);
      const list = wanted.length ? wanted : ALL_METRIC_NAMES;
      queries = list.map((n, i) => {
        // 优先用已知 profile，未知指标自动生成: avg(kaiwu_{name}{})
        const e = METRIC_PROFILES[n] || `avg(kaiwu_${n}{})`;
        return { name: n, expr: e, id: `${n}_${i}`, step: String(kwargs.step) };
      });
    }

    // 一次性多 query 调用
    const body = {
      ...ctx,
      train_task_id: taskId,
      start_time: { timestamp: startISO },
      end_time: { timestamp: endISO },
      queries,
    };
    const data = await api(`${prefix}/GetTrainMetricRange`, body, { session });

    const idx2name = Object.fromEntries(queries.map(q => [q.id, q.name]));
    const exprMap = Object.fromEntries(queries.map(q => [q.name, q.expr]));
    const rows = (data.results || []).map(r => {
      const name = idx2name[r.id] || r.id;
      const allValues = (r.items || []).flatMap(it => (it.values || []).map(v => Number(v.value)));
      if (kwargs['dump-items']) {
        const labels_only = (r.items || []).map(it => it.labels);
        return { name, n: labels_only.length, labels: JSON.stringify(labels_only) };
      }
      if (kwargs.raw) {
        return { name, has_data: allValues.length > 0, points: allValues.length, raw: JSON.stringify(allValues) };
      }
      const nonZero = allValues.filter(v => v !== 0);
      const min = allValues.length ? Math.min(...allValues) : null;
      const max = allValues.length ? Math.max(...allValues) : null;
      const last = allValues.length ? allValues[allValues.length - 1] : null;
      return {
        name,
        has_data: allValues.length > 0 ? (nonZero.length > 0 ? 'yes' : 'all_zero') : 'no',
        points: allValues.length,
        min,
        max,
        last,
        expr: exprMap[name],
      };
    });
    return rows;
  },
};
