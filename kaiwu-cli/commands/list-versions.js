import { api, contextFromArgs, apiPrefix, loadSession } from '../_session.js';

export default {
  name: 'list-versions',
  description: '列出当前实验可用的项目版本（hok1v1 等）',
  example: 'kaiwu list-versions',
  domain: 'tencentarena.com',
  args: [],
  columns: ['project_id', 'project_version', 'experiment_id'],
  run: async (kwargs) => {
    const session = await loadSession();
    const ctx = contextFromArgs(session, kwargs);
    const prefix = apiPrefix(ctx);
    const data = await api(`${prefix}/ListProjectVersion`, ctx, { session });
    return (data.project_versions || []);
  },
};
