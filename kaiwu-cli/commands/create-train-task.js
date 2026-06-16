import { api, withCtx, loadSession } from '../_session.js';

export default {
  name: 'create-train-task',
  description: '创建训练任务（CreateTrainTask）。写操作。默认复用最近 task 的代码包 file.key；dry-run 默认开。',
  example: 'kaiwu create-train-task --name v3-fix-coords --yes',
  domain: 'tencentarena.com',
  args: [
    { name: 'name', type: 'string', required: true, help: '任务名（最长 20 字符）' },
    { name: 'reuse-code-from', type: 'int', required: false, help: '复用某 task 的代码包 file。缺省取最近 1 条 task' },
    { name: 'file-key', type: 'string', default: '', help: '直接指定 file.key（覆盖 --reuse-code-from）' },
    { name: 'file-name', type: 'string', default: '', help: '配合 --file-key 用的 filename' },
    { name: 'file-size', type: 'string', default: '', help: '配合 --file-key 用的 size（字节，字符串）' },
    { name: 'algorithm', type: 'string', default: 'ppo', help: 'ppo / diy' },
    { name: 'training-mode', type: 'string', default: 'distributed', help: 'distributed' },
    { name: 'framework', type: 'string', default: '', help: 'framework 名（默认空）' },
    { name: 'exp-run-time', type: 'int', default: 604800, help: '任务最大运行秒数（默认 7d=604800）' },
    { name: 'desc', type: 'string', default: '', help: '描述（最长 100 字符）' },
    { name: 'pre-train-model-id', type: 'int', required: false, help: 'AiModel id 作为预训练模型起点' },
    { name: 'cluster-config-id', type: 'int', default: 0, help: '集群配置 id（默认 0=系统默认）' },
    { name: 'dry-run', type: 'boolean', default: true, help: '默认 true，不真发；加 --no-dry-run 或 --yes 才真起任务' },
    { name: 'yes', type: 'boolean', default: false, help: '真起训练任务（占用 train_parallel_task 配额）' },
  ],
  columns: ['key', 'value'],
  run: async (kwargs) => {
    const session = await loadSession();
    if (kwargs.name.length > 20) throw new Error(`--name 超过 20 字符上限`);
    if (kwargs.desc && kwargs.desc.length > 100) throw new Error(`--desc 超过 100 字符上限`);

    // 解析 file
    let file;
    if (kwargs['file-key']) {
      if (!kwargs['file-name'] || !kwargs['file-size']) {
        throw new Error('--file-key 模式必须同时给 --file-name 和 --file-size');
      }
      file = {
        bucket: '', region: '',
        key: kwargs['file-key'],
        filename: kwargs['file-name'],
        size: String(kwargs['file-size']),
        type: 'code', id: 0,
      };
    } else {
      const list = await api('/api/v5/Competition/ListTrainTask',
        withCtx(session, { page: { current: 1, size: 50 } }), { session });
      const tasks = list.train_task || [];
      let src;
      if (kwargs['reuse-code-from']) {
        src = tasks.find(t => t.id === kwargs['reuse-code-from']);
        if (!src) throw new Error(`reuse-code-from=${kwargs['reuse-code-from']} 不在最近 50 条任务里`);
      } else {
        src = tasks.find(t => t.file && t.file.key) || tasks[0];
        if (!src) throw new Error('没有任何历史 task 可复用代码包');
      }
      if (!src.file || !src.file.key) {
        throw new Error(`task ${src.id} 没有 file.key 可复用`);
      }
      file = {
        bucket: src.file.bucket || '', region: src.file.region || '',
        key: src.file.key,
        filename: src.file.filename,
        size: String(src.file.size),
        type: src.file.type || 'code',
        id: src.file.id || 0,
      };
    }

    let preTrainModel = null;
    if (kwargs['pre-train-model-id']) {
      // 从 list-ai-models 反查完整字段（id / algorithm / training_mode / file 等）
      const ml = await api('/api/v5/Competition/ListAiModel',
        withCtx(session, { page: { current: 1, size: 50 } }), { session });
      const m = (ml.ai_model || []).find(x => x.id === kwargs['pre-train-model-id']);
      if (!m) throw new Error(`pre-train-model-id=${kwargs['pre-train-model-id']} 不在最近 50 个 ai_model 里`);
      preTrainModel = {
        id: m.id,
        name: m.name,
        algorithm: m.algorithm,
        training_mode: m.training_mode,
        framework: m.framework || '',
        train_step: m.train_step,
        train_time: m.train_time,
        file: m.file,
      };
    }

    const body = withCtx(session, {
      name: kwargs.name,
      algorithm: kwargs.algorithm,
      training_mode: kwargs['training-mode'],
      framework: kwargs.framework,
      exp_run_time: kwargs['exp-run-time'],
      desc: kwargs.desc || '',
      file,
      pre_train_model: preTrainModel,
      cluster_config_id: kwargs['cluster-config-id'],
    });

    // 默认 dry-run，除非 --yes 或 --no-dry-run
    const reallyDo = kwargs.yes || kwargs['dry-run'] === false;
    if (!reallyDo) {
      return [
        { key: 'mode', value: 'DRY-RUN（默认）。加 --yes 才真起任务' },
        { key: 'method', value: 'POST' },
        { key: 'url', value: '/api/v5/Competition/CreateTrainTask' },
        { key: 'body', value: JSON.stringify(body, null, 2) },
        { key: 'reuse_file_from', value: kwargs['reuse-code-from'] || '(最近 task 自动选)' },
        { key: 'file_key', value: file.key },
        { key: 'file_filename', value: file.filename },
        { key: 'file_size_bytes', value: file.size },
      ];
    }

    // 真发
    const data = await api('/api/v5/Competition/CreateTrainTask', body, { session });
    return [
      { key: 'mode', value: 'REAL' },
      { key: 'created_task_id', value: data.id ?? '(查 ListTrainTask)' },
      { key: 'name', value: kwargs.name },
      { key: 'algorithm', value: kwargs.algorithm },
      { key: 'exp_run_time_sec', value: kwargs['exp-run-time'] },
      { key: 'file_key', value: file.key },
      { key: 'response', value: JSON.stringify(data) },
    ];
  },
};
