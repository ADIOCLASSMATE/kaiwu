import { api, loadSession } from '../_session.js';

function fmtDur(sec) {
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return h > 0 ? `${h}h${m}m${s}s` : `${m}m${s}s`;
}

function ts() {
  return new Date().toISOString().replace('T', ' ').slice(0, 19);
}

async function getStatus(session) {
  return api('/api/v5/Competition/GetWebIDE',
    { domain: { id: session.stage_id, type: session.domain_type }, experiment_id: session.experiment_id },
    { session });
}

async function startIDE(session, cur) {
  return api('/api/v5/Competition/StartWebIDE', {
    domain: { id: session.stage_id, type: session.domain_type },
    experiment_id: session.experiment_id,
    competition_team_id: session.team_id,
    cluster_config_id: cur.cluster_config_id,
    project: cur.project,
  }, { session });
}

export default {
  name: 'keep-ide-alive',
  description: '常驻轮询 GetWebIDE，发现 stopped/dead 就自动 StartWebIDE。L2 自动起 env_agent 靠 .vscode/tasks.json runOn:folderOpen，要求 IDE 网页被打开过一次',
  example: 'kaiwu keep-ide-alive --interval 60',
  args: [
    { name: 'interval', type: 'int', default: 60, help: '轮询间隔秒（默认 60）' },
    { name: 'max-loops', type: 'int', default: 0, help: '最多跑多少轮，0=不限（适合测试用 --max-loops 3）' },
    { name: 'log-every-poll', type: 'boolean', default: false, help: '每轮都打 status 行（默认只在变化时打）' },
  ],
  columns: ['key', 'value'],
  run: async (kwargs) => {
    const session = await loadSession();
    const interval = Math.max(10, kwargs.interval);
    const maxLoops = kwargs['max-loops'];
    const verbose = kwargs['log-every-poll'];

    console.error(`[${ts()}] keep-ide-alive 启动 interval=${interval}s max-loops=${maxLoops || 'inf'}`);

    let lastStatus = null;
    let lastDeployAt = null;
    let restartCount = 0;
    let loopCount = 0;

    while (maxLoops === 0 || loopCount < maxLoops) {
      loopCount++;
      try {
        const cur = await getStatus(session);
        const statusChanged = cur.status !== lastStatus;
        const deployChanged = lastDeployAt && cur.deploy_at !== lastDeployAt;

        if (verbose || statusChanged || deployChanged) {
          const tag = deployChanged ? ' [REDEPLOYED]' : (statusChanged ? ' [STATUS CHANGED]' : '');
          console.error(`[${ts()}] status=${cur.status} run_time=${fmtDur(cur.run_time || 0)} deploy_at=${cur.deploy_at}${tag}`);
        }

        if (cur.status !== 'running') {
          console.error(`[${ts()}] IDE not running (status=${cur.status})，触发 StartWebIDE`);
          try {
            const r = await startIDE(session, cur);
            restartCount++;
            console.error(`[${ts()}] StartWebIDE OK (第 ${restartCount} 次自动重启) response=${JSON.stringify(r).slice(0, 200)}`);
            console.error(`[${ts()}] 提醒：env_agent 中转通道断了。打开 IDE 网页一次让 .vscode/tasks.json 自动起 start-env-agent.sh`);
          } catch (e) {
            if (e.code === 1008) {
              console.error(`[${ts()}] StartWebIDE 1008 (web ide 未关闭) — 可能正在重启中，下轮再查`);
            } else {
              console.error(`[${ts()}] StartWebIDE 失败: ${e.code} ${e.message}`);
            }
          }
        } else if (deployChanged) {
          console.error(`[${ts()}] 平台 atomic redeploy 触发 (status 一直 running 但 deploy_at 跳了)。env_agent 进程已随旧 pod 死掉，L2 tasks.json 等下一次 IDE 网页 client 连入时 fire`);
        }

        lastStatus = cur.status;
        lastDeployAt = cur.deploy_at;
      } catch (e) {
        console.error(`[${ts()}] GetWebIDE 失败: ${e.code || ''} ${e.message}`);
      }

      if (maxLoops > 0 && loopCount >= maxLoops) break;
      await new Promise(r => setTimeout(r, interval * 1000));
    }

    return [
      { key: 'loops', value: loopCount },
      { key: 'restarts_triggered', value: restartCount },
      { key: 'last_status', value: lastStatus },
      { key: 'last_deploy_at', value: lastDeployAt },
    ];
  },
};
