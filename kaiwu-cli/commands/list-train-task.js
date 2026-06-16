import { api, withCtx, loadSession } from '../_session.js';

export default {
  name: 'list-train-task',
  description: '列出当前比赛的训练任务（id / name / status / algorithm / 运行时长 / 创建时间 / Grafana monitor 路径）',
  example: 'kaiwu list-train-task --limit 5',
  domain: 'tencentarena.com',
  args: [
    { name: 'limit', type: 'int', default: 10, help: '返回任务数（page.size）' },
    { name: 'page', type: 'int', default: 1, help: '页码（page.current）' },
    { name: 'name', type: 'string', default: '', help: '按任务名筛选（模糊）' },
    { name: 'status', type: 'string', default: '', help: '按状态筛选: running / failed / manual_release / finished 等' },
  ],
  columns: ['id', 'name', 'status', 'algorithm', 'runtime', 'created_at', 'monitor'],
  run: async (kwargs) => {
    const session = await loadSession();
    const body = withCtx(session, {
      page: { current: kwargs.page, size: kwargs.limit },
    });
    if (kwargs.name) body.name = kwargs.name;
    if (kwargs.status) body.status = kwargs.status;
    const data = await api('/api/v5/Competition/ListTrainTask', body, { session });
    const tasks = data.train_task || [];
    return tasks.map(t => ({
      id: t.id,
      name: t.name,
      status: t.status,
      algorithm: t.algorithm,
      runtime: `${Math.floor((t.actual_run_time || 0) / 60)}min`,
      created_at: t.created_at,
      monitor: t.monitor || '',
    }));
  },
};
