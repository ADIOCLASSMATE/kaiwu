import { api, contextFromArgs, apiPrefix, DOMAIN_ARGS, loadSession } from '../_session.js';

export default {
  name: 'list-races',
  description: '列出当前比赛的测评/天梯任务（race），id/name/状态/对战轮次',
  example: 'kaiwu list-races --limit 10',
  domain: 'tencentarena.com',
  args: [
    { name: 'limit', type: 'int', default: 10, help: 'page.size' },
    ...DOMAIN_ARGS,
    { name: 'page', type: 'int', default: 1, help: 'page.current' },
  ],
  columns: ['id', 'name', 'status', 'rule', 'created_at'],
  run: async (kwargs) => {
    const session = await loadSession();
    const ctx = contextFromArgs(session, kwargs);
    const prefix = apiPrefix(ctx);
    const data = await api(`${prefix}/ListRace`,
      { ...ctx,  page: { current: kwargs.page, size: kwargs.limit } }, { session });
    const rows = (data.race || []).map(r => ({
      id: r.id,
      name: r.name,
      status: r.status,
      rule: r.rule,
      created_at: r.created_at,
    }));
    if (!rows.length) {
      return [{ id: '-', name: '(无测评/天梯任务)', status: '-', rule: '-', created_at: '-' }];
    }
    return rows;
  },
};
