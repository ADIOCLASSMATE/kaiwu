import { api, contextFromArgs, apiPrefix, loadSession } from '../_session.js';

export default {
  name: 'team-info',
  description: '查看自己的队伍详情（成员名单、学校、邀请码等）',
  example: 'kaiwu team-info',
  domain: 'tencentarena.com',
  args: [],
  columns: ['key', 'value'],
  run: async (kwargs) => {
    const session = await loadSession();
    const ctx = contextFromArgs(session, kwargs);
    const prefix = apiPrefix(ctx);
    const data = await api(`${prefix}/GetCompetitionTeam`, ctx, { session });
    const t = data.competition_team || {};
    const members = (t.team_members || []).map(m =>
      `${m.real_name || '(未填名)'}/${m.role_code || '-'}/${m.degree?.school || '-'}`
    ).join(' | ');
    return [
      { key: 'team_id', value: t.id },
      { key: 'name', value: t.name },
      { key: 'school', value: t.school_name },
      { key: 'qq', value: t.qq || '(无)' },
      { key: 'competition_id', value: t.competition_id },
      { key: 'root_competition_id', value: t.root_competition_id },
      { key: 'members', value: members },
    ];
  },
};
