import { api, withCtx, loadSession } from '../_session.js';

export default {
  name: 'get-train-task',
  description: '查看单个训练任务详情（含资源用量/算法/环境数/错误信息）。注意：开悟服务端用 uuid 查询，不是 id。',
  example: 'kaiwu get-train-task --uuid ecaaa85e-5a46-40e8-81cb-1566d1e036e6',
  domain: 'tencentarena.com',
  args: [
    { name: 'uuid', type: 'string', required: false, help: '任务 UUID（list-train-task 返回的 uuid 字段）' },
    { name: 'task-id', type: 'int', required: false, help: '任务 ID（自动从 list-train-task 反查 uuid）' },
  ],
  columns: ['key', 'value'],
  run: async (kwargs) => {
    const session = await loadSession();
    let uuid = kwargs.uuid;
    if (!uuid && kwargs['task-id']) {
      const list = await api('/api/v5/Competition/ListTrainTask',
        withCtx(session, { page: { current: 1, size: 50 } }), { session });
      const t = (list.train_task || []).find(x => x.id === kwargs['task-id']);
      if (!t) throw new Error(`task-id=${kwargs['task-id']} 在最近 50 条任务中未找到`);
      uuid = t.uuid;
    }
    if (!uuid) throw new Error('--uuid 或 --task-id 至少给一个');
    const data = await api('/api/v5/Competition/GetTrainTask', withCtx(session, { uuid }), { session });
    const t = data.task || {};
    const fmt = (s) => `${Math.floor((s || 0) / 60)}min`;
    return [
      { key: 'id', value: t.id },
      { key: 'uuid', value: t.uuid },
      { key: 'name', value: t.name },
      { key: 'status', value: t.status },
      { key: 'error_message', value: t.error_message || '(无)' },
      { key: 'algorithm', value: t.algorithm },
      { key: 'creator', value: `${t.creator_name} (id=${t.creator_id})` },
      { key: 'team', value: `${t.team_name} (id=${t.team_id})` },
      { key: 'created_at', value: t.created_at },
      { key: 'start_time', value: t.start_time },
      { key: 'end_time', value: t.end_time || '(未结束)' },
      { key: 'actual_run_time', value: fmt(t.actual_run_time) },
      { key: 'exp_run_time_max', value: fmt(t.exp_run_time) },
      { key: 'monitor', value: t.monitor || '' },
      { key: 'project_version', value: `${t.project?.code} ${t.project?.version}` },
      { key: 'desc', value: t.desc || '(无)' },
      { key: 'pre_train_model', value: t.pre_train_model ? JSON.stringify(t.pre_train_model) : '(无)' },
      { key: 'code_file_key', value: t.file?.key || '(无)' },
      { key: 'code_file_size', value: t.file?.size || '0' },
    ];
  },
};
