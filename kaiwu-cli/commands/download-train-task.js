import { api, contextFromArgs, apiPrefix, DOMAIN_ARGS, loadSession } from '../_session.js';
import { authDownloadAndSave } from '../_download_util.js';

export default {
  name: 'download-train-task',
  description: '下载训练任务的代码包 zip（list-train-task 里那个 task 当时上传的代码）',
  example: 'kaiwu download-train-task --task-id 145415 --output ./r8-code.zip',
  domain: 'tencentarena.com',
  args: [
    { name: 'task-id', type: 'int', required: true, help: '训练任务 id' },
    ...DOMAIN_ARGS,
    { name: 'output', type: 'string', default: '', help: '本地保存路径，缺省取服务端 filename' },
  ],
  columns: ['task_id', 'name', 'size_mb', 'output'],
  run: async (kwargs) => {
    const session = await loadSession();
    const ctx = contextFromArgs(session, kwargs);
    const prefix = apiPrefix(ctx);
    const taskId = kwargs['task-id'];
    let task = null;
    for (let p = 1; p <= 50 && !task; p++) {
      const list = await api(`${prefix}/ListTrainTask`,
        { ...ctx,  page: { current: p, size: 50 } }, { session });
      const items = list.train_task || [];
      task = items.find(t => t.id === taskId);
      if (items.length < 50) break;
    }
    if (!task) throw new Error(`train_task id=${taskId} 没在最近 2500 条命中`);
    const file = task.file;
    if (!file?.key) throw new Error(`train_task id=${taskId} 没 file.key`);
    const out = kwargs.output || file.filename || `train_task_${taskId}.zip`;
    const r = await authDownloadAndSave(session, ctx, file.key,
      { type: 'train_task', id: taskId }, out);
    return [{ task_id: taskId, name: task.name, size_mb: r.size_mb, output: r.output }];
  },
};
