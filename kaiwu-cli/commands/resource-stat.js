import { api, withCtx, loadSession } from '../_session.js';

export default {
  name: 'resource-stat',
  description: '所有模块的资源配额统计（train/battle/dev/ai_model 全部模块）',
  example: 'kaiwu resource-stat',
  domain: 'tencentarena.com',
  args: [],
  columns: ['module', 'key', 'quota'],
  run: async () => {
    const session = await loadSession();
    const data = await api('/api/v5/Competition/GetResourceStat', withCtx(session), { session });
    const rows = [];
    const pack = (data.stat?.pack || [])[0]?.resource_module || {};
    for (const [mod, cfg] of Object.entries(pack)) {
      for (const r of (cfg.resource || [])) {
        rows.push({ module: mod, key: r.key, quota: r.quota });
      }
    }
    return rows;
  },
};
