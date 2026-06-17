import { api, contextFromArgs, apiPrefix, DOMAIN_ARGS, loadSession } from '../_session.js';

export default {
  name: 'list-games',
  description: '列对战任务下的单局对战（含每局 winner=A/B、状态、双方击杀/死亡/KDA）',
  example: 'kaiwu list-games --battle-task-id 490843 --limit 10',
  domain: 'tencentarena.com',
  args: [
    { name: 'battle-task-id', type: 'int', required: true, help: 'battle_task id（list-battle-tasks 看到的）' },
    ...DOMAIN_ARGS,
    { name: 'limit', type: 'int', default: 10, help: 'page.size' },
    { name: 'page', type: 'int', default: 1, help: 'page.current' },
  ],
  columns: ['id', 'name', 'status', 'winner', 'a_kill', 'a_dead', 'a_lineup', 'b_kill', 'b_dead', 'b_lineup'],
  run: async (kwargs) => {
    const session = await loadSession();
    const ctx = contextFromArgs(session, kwargs);
    const prefix = apiPrefix(ctx);
    const data = await api(`${prefix}/ListGame`,
      { ...ctx,  battle_task_id: kwargs['battle-task-id'], page: { current: kwargs.page, size: kwargs.limit } },
      { session });
    const games = data.game || [];
    return games.map(g => {
      const ci = (idx) => {
        const c = (g.camp || [])[idx];
        if (!c) return {};
        let info = {};
        try { info = JSON.parse(c.end_info || '{}'); } catch {}
        return { kill: info.kill_cnt, dead: info.dead_cnt, lineup: info.lineup_code };
      };
      const a = ci(0), b = ci(1);
      return {
        id: g.id,
        name: g.name,
        status: g.status,
        winner: g.result_value,
        a_kill: a.kill, a_dead: a.dead, a_lineup: a.lineup,
        b_kill: b.kill, b_dead: b.dead, b_lineup: b.lineup,
      };
    });
  },
};
