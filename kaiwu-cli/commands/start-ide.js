import { api, contextFromArgs, apiPrefix, DOMAIN_ARGS, loadSession } from '../_session.js';

export default {
  name: 'start-ide',
  description: '启动（或重启）WebIDE 容器。复用当前 cluster_config_id + project 字段，IDE running 时会返 1008',
  example: 'kaiwu start-ide --yes',
  args: [
    { name: 'cluster-config-id', type: 'int', help: '集群配置 id（不传则从当前 GetWebIDE 取）' },
    ...DOMAIN_ARGS,
    { name: 'dry-run', type: 'boolean', default: false, help: '只打印 body 不真发' },
    { name: 'yes', type: 'boolean', default: false, help: '跳过二次确认' },
  ],
  columns: ['key', 'value'],
  run: async (kwargs) => {
    const session = await loadSession();
    const ctx = contextFromArgs(session, kwargs);
    const prefix = apiPrefix(ctx);
    const cur = await api(`${prefix}/GetWebIDE`,
      { domain: { id: session.stage_id, type: session.domain_type }, experiment_id: session.experiment_id },
      { session });
    const body = {
      domain: { id: session.stage_id, type: session.domain_type },
      experiment_id: session.experiment_id,
      competition_team_id: session.team_id,
      cluster_config_id: kwargs['cluster-config-id'] || cur.cluster_config_id,
      project: cur.project,
    };
    if (kwargs['dry-run']) {
      return [
        { key: 'method', value: 'POST' },
        { key: 'url', value: `${prefix}/StartWebIDE` },
        { key: 'body', value: JSON.stringify(body) },
        { key: 'current_status', value: cur.status },
        { key: 'note', value: '加 --yes 真发请求' },
      ];
    }
    if (!kwargs.yes && cur.status === 'running') {
      return [
        { key: 'aborted', value: `IDE 已 running (id=${cur.id} run_time=${cur.run_time}s)，无需启动` },
        { key: 'note', value: '加 --yes 强制下发（会得 1008 错误）' },
      ];
    }
    try {
      const data = await api(`${prefix}/StartWebIDE`, body, { session });
      return [
        { key: 'result', value: 'started' },
        { key: 'old_status', value: cur.status },
        { key: 'response', value: JSON.stringify(data) },
      ];
    } catch (e) {
      if (e.code === 1008) {
        return [
          { key: 'noop', value: 'IDE 已运行 (1008 web ide 未关闭)' },
          { key: 'current_status', value: cur.status },
        ];
      }
      throw e;
    }
  },
};
