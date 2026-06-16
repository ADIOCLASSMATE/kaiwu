import { api, withCtx, loadSession } from '../_session.js';

export default {
  name: 'stop-battle-task',
  description: '停止对战任务（terminate）。写操作，会立即终止 running 的 BattleTask。',
  example: 'kaiwu stop-battle-task --id 490872',
  domain: 'tencentarena.com',
  args: [
    { name: 'id', type: 'int', required: true, help: 'battle_task id' },
  ],
  columns: ['key', 'value'],
  run: async (kwargs) => {
    const session = await loadSession();
    await api('/api/v5/Competition/StopBattleTask', withCtx(session, { id: kwargs.id }), { session });
    return [
      { key: 'battle_task_id', value: kwargs.id },
      { key: 'result', value: 'terminated' },
    ];
  },
};
