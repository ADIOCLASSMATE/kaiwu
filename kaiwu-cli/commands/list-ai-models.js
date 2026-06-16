import { api, withCtx, loadSession } from '../_session.js';

export default {
  name: 'list-ai-models',
  description: '列出用户模型库（永久存储的提交模型，区别于 list-models 的训练任务临时 checkpoint）',
  example: 'kaiwu list-ai-models --limit 10',
  domain: 'tencentarena.com',
  args: [
    { name: 'limit', type: 'int', default: 10, help: 'page.size' },
    { name: 'page', type: 'int', default: 1, help: 'page.current' },
    { name: 'name', type: 'string', default: '', help: '名称模糊筛选' },
  ],
  columns: ['id', 'name', 'status', 'algorithm', 'train_step', 'train_time', 'created_at'],
  run: async (kwargs) => {
    const session = await loadSession();
    const body = withCtx(session, { page: { current: kwargs.page, size: kwargs.limit } });
    if (kwargs.name) body.name = kwargs.name;
    const data = await api('/api/v5/Competition/ListAiModel', body, { session });
    return (data.ai_model || []).map(m => ({
      id: m.id,
      name: m.name,
      status: m.status,
      algorithm: m.algorithm,
      train_step: m.train_step,
      train_time: `${Math.floor((m.train_time || 0) / 60)}min`,
      created_at: m.created_at,
    }));
  },
};
