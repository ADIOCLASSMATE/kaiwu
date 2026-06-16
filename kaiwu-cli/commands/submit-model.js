import { api, withCtx, loadSession } from '../_session.js';

export default {
  name: 'submit-model',
  description: '把训练任务的某个 checkpoint 提交到永久模型库（CreateAiModel）。注意：是写操作。',
  example: 'kaiwu submit-model --task-id 145229 --latest',
  domain: 'tencentarena.com',
  args: [
    { name: 'task-id', type: 'int', required: true, help: '训练任务 id（list-train-task 的 id 字段）' },
    { name: 'train-step', type: 'int', required: false, help: '指定 checkpoint 的 train_step；与 --latest 二选一' },
    { name: 'latest', type: 'boolean', default: false, help: '自动选 task 最新 checkpoint（按 train_step 最大）' },
    { name: 'name', type: 'string', default: '', help: '模型名（默认 auto-{taskid}-{step}，最长 20 字符）' },
    { name: 'desc', type: 'string', default: '', help: '描述（最长 100 字符）' },
    { name: 'share', type: 'string', default: 'team', help: '共享范围: user / team / experiment' },
    { name: 'training-mode', type: 'string', default: 'distributed', help: 'distributed / local 等' },
    { name: 'algorithm', type: 'string', default: 'ppo', help: '算法名' },
    { name: 'dry-run', type: 'boolean', default: false, help: '只打印 body 不真发' },
  ],
  columns: ['key', 'value'],
  run: async (kwargs) => {
    const session = await loadSession();
    const taskId = kwargs['task-id'];
    if (!kwargs.latest && kwargs['train-step'] === undefined) {
      throw new Error('--train-step 或 --latest 至少给一个');
    }
    if (!['user', 'team', 'experiment'].includes(kwargs.share)) {
      throw new Error(`--share 必须是 user/team/experiment，收到: ${kwargs.share}`);
    }

    // 拿 checkpoint
    const modelsResp = await api('/api/v5/Competition/ListTrainAiModel',
      withCtx(session, { train_task_id: taskId }), { session });
    const ckpts = modelsResp.ai_models || [];
    if (!ckpts.length) throw new Error(`task ${taskId} 没有 checkpoint，无法提交`);
    let ck;
    if (kwargs.latest) {
      ck = ckpts.reduce((a, b) => (b.train_step > a.train_step ? b : a));
    } else {
      ck = ckpts.find(c => c.train_step === kwargs['train-step']);
      if (!ck) throw new Error(`task ${taskId} 没有 train_step=${kwargs['train-step']} 的 checkpoint。可选: ${ckpts.map(c => c.train_step).join(', ')}`);
    }

    const step = ck.train_step;
    const name = kwargs.name || `auto-${taskId}-${step}`;
    if (name.length > 20) throw new Error(`模型名 "${name}" 超过 20 字符上限`);
    if (kwargs.desc && kwargs.desc.length > 100) throw new Error(`描述超过 100 字符上限`);

    // 服务端要的 filename 是 `hok1v1-ppo-{step}.zip` 不是 list-models 返回的完整名
    const projectCode = 'hok1v1';
    const file = {
      bucket: '',
      region: '',
      key: ck.file.key,
      filename: `${projectCode}-${kwargs.algorithm}-${step}.zip`,
      size: String(ck.file.size),
      type: '',
      updated_at: null,
      id: 0,
    };

    const body = {
      algorithm: kwargs.algorithm,
      training_mode: kwargs['training-mode'],
      train_task_id: taskId,
      train_time: ck.train_time,
      train_step: step,
      name,
      desc: kwargs.desc || '',
      share_type: kwargs.share,
      file,
      ...withCtx(session),
    };

    if (kwargs['dry-run']) {
      return [
        { key: 'method', value: 'POST' },
        { key: 'url', value: '/api/v5/Competition/CreateAiModel' },
        { key: 'body', value: JSON.stringify(body, null, 2) },
        { key: 'note', value: '加 --no-dry-run 真提交' },
      ];
    }

    const resp = await api('/api/v5/Competition/CreateAiModel', body, { session });
    return [
      { key: 'ai_model_id', value: resp.id },
      { key: 'name', value: name },
      { key: 'train_task_id', value: taskId },
      { key: 'train_step', value: step },
      { key: 'train_time_min', value: Math.floor((ck.train_time || 0) / 60) },
      { key: 'share_type', value: kwargs.share },
      { key: 'algorithm', value: kwargs.algorithm },
      { key: 'training_mode', value: kwargs['training-mode'] },
      { key: 'file_key', value: ck.file.key },
    ];
  },
};
