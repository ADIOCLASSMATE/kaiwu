import { api, contextFromArgs, apiPrefix, DOMAIN_ARGS, loadSession } from '../_session.js';

export default {
  name: 'create-battle-task',
  description: '创建评估对战任务（CreateBattleTask）。两个 ai_model 互打 N round（每 round 通常 8 局，共 round*8 局）。写操作，dry-run 默认开。',
  example: 'kaiwu create-battle-task --name eval-v3-vs-baseline --camp-a 254214 --camp-b 254190 --round 5 --yes',
  domain: 'tencentarena.com',
  args: [
    { name: 'name', type: 'string', required: true, help: '对战任务名' },
    ...DOMAIN_ARGS,
    { name: 'camp-a', type: 'int', required: true, help: 'A 方 ai_model_id（list-ai-models 的 id）' },
    { name: 'camp-b', type: 'int', required: true, help: 'B 方 ai_model_id' },
    { name: 'round', type: 'int', default: 5, help: '对战轮次（每轮通常含 8 局）' },
    { name: 'share', type: 'string', default: 'team', help: 'user / team / experiment' },
    { name: 'hero-a', type: 'int', required: false, help: 'A 方英雄 id（112=鲁班，133=狄仁杰，缺省双阵容）' },
    { name: 'hero-b', type: 'int', required: false, help: 'B 方英雄 id' },
    { name: 'summoner-a', type: 'int', required: false, help: 'A 方召唤师技能 id（80115=闪现）' },
    { name: 'summoner-b', type: 'int', required: false, help: 'B 方召唤师技能 id' },
    { name: 'dry-run', type: 'boolean', default: true, help: '默认开，加 --no-dry-run 或 --yes 才真创建' },
    { name: 'yes', type: 'boolean', default: false, help: '真创建（消耗 battle_max_game 配额）' },
  ],
  columns: ['key', 'value'],
  run: async (kwargs) => {
    const session = await loadSession();
    const ctx = contextFromArgs(session, kwargs);
    const prefix = apiPrefix(ctx);
    if (!['user', 'team', 'experiment'].includes(kwargs.share)) {
      throw new Error(`--share 必须是 user/team/experiment，收到: ${kwargs.share}`);
    }
    const buildCamp = (model, hero, summoner) => {
      const c = { ai_model_id: model };
      if (hero !== undefined) c.hero_id = hero;
      if (summoner !== undefined) c.summoner_skill_id = summoner;
      return c;
    };
    const camp = [
      buildCamp(kwargs['camp-a'], kwargs['hero-a'], kwargs['summoner-a']),
      buildCamp(kwargs['camp-b'], kwargs['hero-b'], kwargs['summoner-b']),
    ];
    const body = { ...ctx, 
      name: kwargs.name,
      share_type: kwargs.share,
      round: kwargs.round,
      camp,
    };
    const reallyDo = kwargs.yes || kwargs['dry-run'] === false;
    if (!reallyDo) {
      return [
        { key: 'mode', value: 'DRY-RUN（默认）。加 --yes 真创建' },
        { key: 'method', value: 'POST' },
        { key: 'url', value: `${prefix}/CreateBattleTask` },
        { key: 'body', value: JSON.stringify(body, null, 2) },
        { key: 'estimated_games', value: kwargs.round * 8 },
      ];
    }
    const data = await api(`${prefix}/CreateBattleTask`, body, { session });
    return [
      { key: 'mode', value: 'REAL' },
      { key: 'battle_task_id', value: data.id },
      { key: 'name', value: kwargs.name },
      { key: 'camp_a_model', value: kwargs['camp-a'] },
      { key: 'camp_b_model', value: kwargs['camp-b'] },
      { key: 'round', value: kwargs.round },
      { key: 'share_type', value: kwargs.share },
    ];
  },
};
