import { api, withCtx, loadSession } from '../_session.js';

export default {
  name: 'list-battle-tasks',
  description: '列出对战任务（评估/天梯产生的具体对局任务）',
  example: 'kaiwu list-battle-tasks --limit 10',
  domain: 'tencentarena.com',
  args: [
    { name: 'owner-type', type: 'string', default: 'battle', help: 'owner.type，默认 battle 看用户自己创建的' },
    { name: 'owner-id', type: 'int', default: 0, help: 'owner.id，battle 域默认 0' },
    { name: 'limit', type: 'int', default: 10, help: 'page.size' },
    { name: 'page', type: 'int', default: 1, help: 'page.current' },
  ],
  columns: ['id', 'name', 'status', 'round', 'game_count', 'created_at'],
  run: async (kwargs) => {
    const session = await loadSession();
    const data = await api('/api/v5/Competition/ListBattleTask',
      withCtx(session, {
        owner: { type: kwargs['owner-type'], id: kwargs['owner-id'] },
        page: { current: kwargs.page, size: kwargs.limit },
      }), { session });
    const rows = (data.battle_task || []).map(b => ({
      id: b.id,
      name: b.name,
      status: b.status,
      round: b.round,
      game_count: b.game_count,
      created_at: b.created_at,
    }));
    if (!rows.length) {
      return [{ id: '-', name: `(无对战, owner=${kwargs['owner-type']}/${kwargs['owner-id']})`, status: '-', round: '-', game_count: '-', created_at: '-' }];
    }
    return rows;
  },
};
