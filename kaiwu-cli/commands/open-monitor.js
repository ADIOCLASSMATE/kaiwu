import { api, withCtx, loadSession } from '../_session.js';
import { execSync } from 'node:child_process';

export default {
  name: 'open-monitor',
  description: '在浏览器打开训练任务的 Grafana 监控面板（reward/win_rate/tower_hp/kill/death）',
  example: 'kaiwu open-monitor --task-id 145229',
  domain: 'tencentarena.com',
  args: [
    { name: 'task-id', type: 'int', required: false, help: '训练任务 id（默认取最新 running 的）' },
    { name: 'in-automation', type: 'boolean', default: false, help: '在 opencli automation window 打开（默认走系统浏览器）' },
  ],
  columns: ['key', 'value'],
  run: async (kwargs) => {
    const session = await loadSession();
    const list = await api('/api/v5/Competition/ListTrainTask',
      withCtx(session, { page: { current: 1, size: 50 } }), { session });
    const tasks = list.train_task || [];
    let t;
    if (kwargs['task-id']) {
      t = tasks.find(x => x.id === kwargs['task-id']);
      if (!t) throw new Error(`task-id=${kwargs['task-id']} 不在最近 50 条任务里`);
    } else {
      t = tasks.find(x => x.status === 'running') || tasks[0];
      if (!t) throw new Error('没有任务可打开');
    }
    if (!t.monitor) throw new Error(`task ${t.id} 没有 monitor URL`);
    const url = 'https://tencentarena.com' + t.monitor + `&var-train_task=${t.id}`;
    if (kwargs['in-automation']) {
      execSync(`opencli browser open ${JSON.stringify(url)}`, { stdio: 'inherit' });
    } else {
      execSync(`open ${JSON.stringify(url)}`);
    }
    return [
      { key: 'task_id', value: t.id },
      { key: 'task_name', value: t.name },
      { key: 'status', value: t.status },
      { key: 'url', value: url },
    ];
  },
};
