import { api, withCtx, loadSession } from '../_session.js';
import { authDownloadAndSave } from '../_download_util.js';

export default {
  name: 'download-game',
  description: '下载评估对战中某一局的录像（abs）或日志（log）',
  example: 'kaiwu download-game --battle-task-id 491126 --game-id 16722075 --kind abs --output ./game-rec.zip',
  domain: 'tencentarena.com',
  args: [
    { name: 'battle-task-id', type: 'int', required: true, help: '对战任务 id（list-battle-tasks）' },
    { name: 'game-id', type: 'int', required: true, help: '单局 id（list-games）' },
    { name: 'kind', type: 'string', default: 'abs', help: '"abs" 录像 或 "log" 日志' },
    { name: 'output', type: 'string', default: '', help: '本地保存路径，缺省取服务端 filename' },
  ],
  columns: ['game_id', 'kind', 'size_mb', 'output'],
  run: async (kwargs) => {
    const session = await loadSession();
    const battleTaskId = kwargs['battle-task-id'];
    const gameId = kwargs['game-id'];
    const kind = String(kwargs.kind || 'abs').toLowerCase();
    if (kind !== 'abs' && kind !== 'log') throw new Error('--kind 只能是 abs 或 log');

    let game = null;
    for (let p = 1; p <= 50 && !game; p++) {
      const list = await api('/api/v5/Competition/ListGame',
        withCtx(session, { battle_task_id: battleTaskId, page: { current: p, size: 50 } }), { session });
      const items = list.game || [];
      game = items.find(g => g.id === gameId);
      if (items.length < 50) break;
    }
    if (!game) throw new Error(`game id=${gameId} 不在 battle_task=${battleTaskId} 里`);
    const file = (game.file || []).find(f => f.type === kind);
    if (!file?.key) throw new Error(`game id=${gameId} 没 type=${kind} 的 file`);
    const out = kwargs.output || file.filename || `game_${gameId}_${kind}.zip`;
    const r = await authDownloadAndSave(session, file.key,
      { type: 'game', id: gameId }, out);
    return [{ game_id: gameId, kind, size_mb: r.size_mb, output: r.output }];
  },
};
