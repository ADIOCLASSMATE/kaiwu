import { api, loadSession } from '../_session.js';

export default {
  name: 'experiment',
  description: '查看当前实验配置（项目版本 / 算法白名单 / 资源部署模板 / 评估配置）',
  example: 'kaiwu experiment',
  domain: 'tencentarena.com',
  args: [
    { name: 'full', type: 'boolean', default: false, help: '展开完整 JSON' },
  ],
  columns: ['key', 'value'],
  run: async (kwargs) => {
    const session = await loadSession();
    const data = await api('/api/v5/Competition/GetExperimentHistory',
      { domain: { id: session.stage_id, type: session.domain_type }, experiment_id: session.experiment_id },
      { session });
    const eh = data.experiment_history || {};
    const pc = eh.project_config || {};
    if (kwargs.full) {
      return [{ key: 'json', value: JSON.stringify(eh, null, 2) }];
    }
    const dm = pc.deploy_modules?.deploy_configs || {};
    const summarizeMods = (m) => (m?.modules || []).map(x => `${x.code}(cpu=${x.cpu}, gpu=${x.gpu}, mem=${x.memory})`).join(' | ');
    const algos = (pc.framework || []).flatMap(f => f.algorithm).join(', ');
    return [
      { key: 'experiment_id', value: eh.experiment_id },
      { key: 'project', value: `${pc.code} v${eh.project_version}` },
      { key: 'project_name', value: pc.name },
      { key: 'algorithms', value: algos },
      { key: 'agent_num', value: pc.agent_num },
      { key: 'deploy.battle', value: summarizeMods(dm.battle) },
      { key: 'deploy.dev', value: summarizeMods(dm.dev) },
      { key: 'deploy.train_cluster', value: summarizeMods(dm.train_cluster) },
      { key: 'parallel_env_per_aisrv', value: dm.train_cluster?.train_config?.parallel_env_per_aisrv ?? '-' },
      { key: 'race_types', value: (pc.race_configs?.race_types || []).map(r => r.name).join(', ') },
      { key: 'created_at', value: eh.created_at },
    ];
  },
};
