import { api, contextFromArgs, apiPrefix, DOMAIN_ARGS, loadSession } from '../_session.js';

export default {
  name: 'resource-balance',
  description: '查看训练资源配额（CPU/GPU/并行任务数/最大训练时长）',
  example: 'kaiwu resource-balance',
  domain: 'tencentarena.com',
  args: [
    { name: 'module', type: 'string', default: 'train', help: '资源模块: train / dev / battle' },
    ...DOMAIN_ARGS,
  ],
  columns: ['key', 'quota', 'used'],
  run: async (kwargs) => {
    const session = await loadSession();
    const ctx = contextFromArgs(session, kwargs);
    const prefix = apiPrefix(ctx);
    const data = await api(`${prefix}/GetResourceBalance`,
      { ...ctx,  resource_module: kwargs.module }, { session });
    const stat = data.stat || {};
    const packResource = ((stat.pack || [])[0]?.resource_module?.[kwargs.module]?.resource) || [];
    const usedResource = ((stat.used || [])[0]?.resource_module?.[kwargs.module]?.resource) || [];
    const usedMap = Object.fromEntries(usedResource.map(r => [r.key, r.quota]));
    return packResource.map(r => ({
      key: r.key,
      quota: r.quota,
      used: usedMap[r.key] ?? '0',
    }));
  },
};
