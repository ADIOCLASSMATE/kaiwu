import { api, withCtx, loadSession } from '../_session.js';

// 解析 camp.end_info（聚合后的 JSON 字符串），提取 win/kill/death/kda 等核心评估数据
function parseEndInfo(s) {
  if (!s) return null;
  try {
    const j = JSON.parse(s);
    const out = {};
    // 顶层（如果是 game-level end_info）
    if (typeof j.win_cnt !== 'undefined') {
      Object.assign(out, {
        win_cnt: j.win_cnt, kill_cnt: j.kill_cnt, dead_cnt: j.dead_cnt,
        kda: j.kda, frame: j.frame,
        money: j.money, hurt_to_hero: j.hurt_to_hero, hurt_by_hero: j.hurt_by_hero,
        destroy_tower_cnt: j.destroy_tower_cnt,
        lineup_code: j.lineup_code,
      });
      return out;
    }
    // battle_task 聚合层（如 'all' / 'Luban' / 'DiRenjie'）
    const breakdown = {};
    for (const [k, v] of Object.entries(j)) {
      if (v && v.end_info && v.game_count !== undefined) {
        breakdown[k] = {
          game_count: v.game_count,
          win_cnt: v.end_info.win_cnt,
          kda: v.end_info.kda,
          kill_cnt: v.end_info.kill_cnt,
          dead_cnt: v.end_info.dead_cnt,
          destroy_tower: v.end_info.destroy_tower_cnt,
          status: v.game_status_count,
        };
      }
    }
    return breakdown;
  } catch (e) {
    return { _parse_error: e.message };
  }
}

export default {
  name: 'get-battle-task',
  description: '看对战任务详情：双方模型 / 胜场 / KDA / 击杀 / 推塔（解析 camp.end_info 聚合数据）',
  example: 'kaiwu get-battle-task --id 490843',
  domain: 'tencentarena.com',
  args: [
    { name: 'id', type: 'int', required: true, help: 'battle_task id' },
  ],
  columns: ['key', 'value'],
  run: async (kwargs) => {
    const session = await loadSession();
    const data = await api('/api/v5/Competition/GetBattleTask', withCtx(session, { id: kwargs.id }), { session });
    const bt = data.battle_task || {};
    const camps = bt.camp || [];
    const summary = camps.map((c, i) => {
      const stats = parseEndInfo(c.end_info);
      const model = c.battle_model || {};
      return {
        idx: i, code: c.code, type: c.type,
        ai_model_id: c.ai_model_id, ai_model_name: model.name,
        train_step: model.train_step, algorithm: model.algorithm,
        stats,
      };
    });
    return [
      { key: 'id', value: bt.id },
      { key: 'name', value: bt.name },
      { key: 'status', value: bt.status },
      { key: 'round', value: bt.round },
      { key: 'game_count', value: bt.game_count },
      { key: 'creator', value: `${bt.creator_name} (id=${bt.creator_id})` },
      { key: 'team', value: `${bt.team_name}` },
      { key: 'share', value: bt.share ? `${bt.share.type}/${bt.share.id}` : '-' },
      { key: 'created_at', value: bt.created_at },
      { key: 'project', value: `${bt.project?.code} ${bt.project?.version}` },
      { key: 'camp_summary', value: JSON.stringify(summary, null, 2) },
    ];
  },
};
