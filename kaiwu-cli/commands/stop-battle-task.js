import { api, contextFromArgs, apiPrefix, DOMAIN_ARGS, loadSession } from '../_session.js';

export default {
  name: 'stop-battle-task',
  description: '停止对战任务（terminate）。写操作，会立即终止 running 的 BattleTask。',
  example: 'kaiwu stop-battle-task --id 490872',
  domain: 'tencentarena.com',
  args: [
    { name: 'id', type: 'int', required: true, help: 'battle_task id' },
    ...DOMAIN_ARGS,
  ],
  columns: ['key', 'value'],
  run: async (kwargs) => {
    const session = await loadSession();
    const ctx = contextFromArgs(session, kwargs);
    const prefix = apiPrefix(ctx);
    await api(`${prefix}/StopBattleTask`, { ...ctx,  id: kwargs.id }, { session });
    return [
      { key: 'battle_task_id', value: kwargs.id },
      { key: 'result', value: 'terminated' },
    ];
  },
};
