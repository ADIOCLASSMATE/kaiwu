import { api, withCtx, loadSession } from '../_session.js';

export default {
  name: 'list-versions',
  description: '列出当前实验可用的项目版本（hok1v1 等）',
  example: 'kaiwu list-versions',
  domain: 'tencentarena.com',
  args: [],
  columns: ['project_id', 'project_version', 'experiment_id'],
  run: async () => {
    const session = await loadSession();
    const data = await api('/api/v5/Competition/ListProjectVersion', withCtx(session), { session });
    return (data.project_versions || []);
  },
};
