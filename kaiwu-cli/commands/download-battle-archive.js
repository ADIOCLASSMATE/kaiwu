import { api, contextFromArgs, apiPrefix, DOMAIN_ARGS, loadSession } from '../_session.js';
import { authDownloadAndSave } from '../_download_util.js';

export default {
  name: 'download-battle-archive',
  description: '下载整个对战任务的全部录像或日志打包 zip（异步：先 ArchiveBattleTaskFile 触发后端打包，再轮询 GetBattleTask 拿 file_key，最后 GetAuthDownloadURL）',
  example: 'kaiwu download-battle-archive --battle-task-id 491126 --kind abs --output ./all-rec.zip',
  domain: 'tencentarena.com',
  args: [
    { name: 'battle-task-id', type: 'int', required: true, help: '对战任务 id' },
    ...DOMAIN_ARGS,
    { name: 'kind', type: 'string', default: 'abs', help: '"abs" 全部录像 或 "log" 全部日志' },
    { name: 'output', type: 'string', default: '', help: '本地保存路径' },
    { name: 'poll-interval', type: 'int', default: 3, help: '轮询间隔秒' },
    { name: 'poll-timeout', type: 'int', default: 120, help: '轮询超时秒' },
  ],
  columns: ['battle_task_id', 'kind', 'archive_status', 'size_mb', 'output'],
  run: async (kwargs) => {
    const session = await loadSession();
    const ctx = contextFromArgs(session, kwargs);
    const prefix = apiPrefix(ctx);
    const battleTaskId = kwargs['battle-task-id'];
    const kind = String(kwargs.kind || 'abs').toLowerCase();
    if (kind !== 'abs' && kind !== 'log') throw new Error('--kind 只能是 abs 或 log');

    // 1. 触发 archive
    await api(`${prefix}/ArchiveBattleTaskFile`,
      { ...ctx,  id: battleTaskId, file_type: kind }, { session });

    // 2. 轮询 GetBattleTask 直到 file_key 出现
    const deadline = Date.now() + kwargs['poll-timeout'] * 1000;
    let file = null;
    let lastStatus = 'unknown';
    while (Date.now() < deadline && !file) {
      const t = await api(`${prefix}/GetBattleTask`,
        { ...ctx,  id: battleTaskId }, { session });
      const archive = t?.battle_task?.archive_file || t?.archive_file || {};
      const slot = archive[kind] || archive[`${kind}_file`];
      lastStatus = slot?.status || archive[`${kind}_status`] || 'pending';
      if (slot?.file?.key) { file = slot.file; break; }
      // alt schema：t.archive_abs_file / t.archive_log_file
      const altKey = `archive_${kind}_file`;
      const alt = t?.battle_task?.[altKey] || t?.[altKey];
      if (alt?.key) { file = alt; break; }
      await new Promise(r => setTimeout(r, kwargs['poll-interval'] * 1000));
    }
    if (!file) throw new Error(`轮询超时 ${kwargs['poll-timeout']}s，archive 未就绪 (last_status=${lastStatus})`);

    // 3. 落盘
    const out = kwargs.output || file.filename || `battle_${battleTaskId}_${kind}.zip`;
    const r = await authDownloadAndSave(session, ctx, file.key,
      { type: 'game', id: 0 }, out);
    return [{
      battle_task_id: battleTaskId, kind, archive_status: lastStatus,
      size_mb: r.size_mb, output: r.output,
    }];
  },
};
