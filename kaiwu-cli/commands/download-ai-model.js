import { api, contextFromArgs, apiPrefix, DOMAIN_ARGS, loadSession } from '../_session.js';
import { authDownloadAndSave } from '../_download_util.js';

export default {
  name: 'download-ai-model',
  description: '从模型库（list-ai-models）下载 ai_model zip 到本地',
  example: 'kaiwu download-ai-model --id 254368 --output ./r8.zip',
  domain: 'tencentarena.com',
  args: [
    { name: 'id', type: 'int', required: true, help: 'ai_model_id（list-ai-models 的 id）' },
    ...DOMAIN_ARGS,
    { name: 'output', type: 'string', default: '', help: '本地保存路径，缺省取服务端 filename 落到 cwd' },
  ],
  columns: ['id', 'name', 'train_step', 'size_mb', 'output'],
  run: async (kwargs) => {
    const session = await loadSession();
    const ctx = contextFromArgs(session, kwargs);
    const prefix = apiPrefix(ctx);
    const id = kwargs.id;
    let model = null;
    for (let p = 1; p <= 50 && !model; p++) {
      const list = await api(`${prefix}/ListAiModel`,
        { ...ctx,  page: { current: p, size: 50 } }, { session });
      const items = list.ai_model || [];
      model = items.find(m => m.id === id);
      if (items.length < 50) break;
    }
    if (!model) throw new Error(`ai_model id=${id} 没在最近 2500 条命中`);
    const file = (model.file || [])[0];
    if (!file?.key) throw new Error(`ai_model id=${id} 没 file.key`);
    const out = kwargs.output || file.filename || `ai_model_${id}.zip`;
    const r = await authDownloadAndSave(session, ctx, file.key,
      { type: 'ai_model', id }, out);
    return [{ id, name: model.name, train_step: model.train_step, size_mb: r.size_mb, output: r.output }];
  },
};
