import { api, withCtx, loadSession } from '../_session.js';

export default {
  name: 'release-train-task',
  description: '释放（停止）训练任务。!注意!这是写操作，会真停掉 running 的任务。建议先用 --dry-run 验证 task-id。',
  example: 'kaiwu release-train-task --task-id 145229',
  domain: 'tencentarena.com',
  args: [
    { name: 'task-id', type: 'int', required: true, help: '训练任务 ID（list-train-task 的 id 字段）' },
    { name: 'dry-run', type: 'boolean', default: false, help: '只打印将要发的请求，不真发' },
    { name: 'yes', type: 'boolean', default: false, help: '跳过二次确认（脚本里用）' },
  ],
  columns: ['key', 'value'],
  run: async (kwargs) => {
    const session = await loadSession();
    const taskId = kwargs['task-id'];
    if (kwargs['dry-run']) {
      return [
        { key: 'method', value: 'POST' },
        { key: 'url', value: '/api/v5/Competition/ReleaseTrainTask' },
        { key: 'body', value: JSON.stringify(withCtx(session, { train_task_id: taskId })) },
        { key: 'note', value: '加 --no-dry-run 真发请求' },
      ];
    }
    if (!kwargs.yes) {
      // 先 GetTrainTask 看 status，避免误释放 running 任务
      const list = await api('/api/v5/Competition/ListTrainTask',
        withCtx(session, { page: { current: 1, size: 50 } }), { session });
      const t = (list.train_task || []).find(x => x.id === taskId);
      if (!t) throw new Error(`task-id=${taskId} 不在最近 50 条任务里。确认 id 后加 --yes 强制执行`);
      if (t.status !== 'running') {
        return [
          { key: 'aborted', value: `task ${taskId} 状态=${t.status}，非 running 无需释放（加 --yes 强制下发）` },
          { key: 'task_name', value: t.name },
          { key: 'task_status', value: t.status },
        ];
      }
      // 是 running，提示
      console.error(`# task ${taskId} (${t.name}) 当前 status=running，将真停训练。继续? 加 --yes 跳过此提示。`);
      console.error(`# 已运行 ${Math.floor((t.actual_run_time || 0) / 60)}min`);
      throw new Error('未 --yes 确认，已中止');
    }
    const data = await api('/api/v5/Competition/ReleaseTrainTask',
      withCtx(session, { train_task_id: taskId }), { session });
    return [
      { key: 'task_id', value: taskId },
      { key: 'result', value: 'released' },
      { key: 'response', value: JSON.stringify(data) },
    ];
  },
};
