import { api, contextFromArgs, apiPrefix, loadSession } from '../_session.js';

export default {
  name: 'ide-status',
  description: '查看在线开发 WebIDE 容器状态（id / status / 部署时间 / 已运行时长）',
  example: 'kaiwu ide-status',
  domain: 'tencentarena.com',
  args: [],
  columns: ['key', 'value'],
  run: async (kwargs) => {
    const session = await loadSession();
    const ctx = contextFromArgs(session, kwargs);
    const prefix = apiPrefix(ctx);
    const data = await api(`${prefix}/GetWebIDE`,
      { domain: { id: session.stage_id, type: session.domain_type }, experiment_id: session.experiment_id },
      { session });
    const fmt = (s) => {
      const m = Math.floor(s / 60), h = Math.floor(m / 60);
      return h > 0 ? `${h}h${m % 60}m` : `${m}m${s % 60}s`;
    };
    return [
      { key: 'id', value: data.id },
      { key: 'status', value: data.status },
      { key: 'cluster_config_id', value: data.cluster_config_id },
      { key: 'project_version', value: `${data.project?.code} ${data.project?.version}` },
      { key: 'deploy_at', value: data.deploy_at },
      { key: 'run_time', value: fmt(data.run_time || 0) },
      { key: 'host', value: data.host || '(无)' },
      { key: 'created_at', value: data.created_at },
    ];
  },
};
