import { api, contextFromArgs, apiPrefix, DOMAIN_ARGS, loadSession } from '../_session.js';

export default {
  name: 'list-models',
  description: '列出某个训练任务产出的模型 checkpoint（train_step / train_time / file_key / size）',
  example: 'kaiwu list-models --task-id 145072',
  domain: 'tencentarena.com',
  args: [
    { name: 'task-id', type: 'int', required: true, help: '训练任务 ID（来自 list-train-task 的 id 字段）' },
    ...DOMAIN_ARGS,
  ],
  columns: ['train_step', 'train_time', 'created_at', 'size_mb', 'filename', 'file_key'],
  run: async (kwargs) => {
    const session = await loadSession();
    const ctx = contextFromArgs(session, kwargs);
    const prefix = apiPrefix(ctx);
    const data = await api(`${prefix}/ListTrainAiModel`,
      { ...ctx,  train_task_id: kwargs['task-id'] }, { session });
    const models = data.ai_models || [];
    return models.map(m => ({
      train_step: m.train_step,
      train_time: `${Math.floor((m.train_time || 0) / 60)}min`,
      created_at: m.created_at,
      size_mb: ((parseInt(m.file?.size || '0', 10) / 1024 / 1024)).toFixed(1),
      filename: m.file?.filename,
      file_key: m.file?.key,
    }));
  },
};
