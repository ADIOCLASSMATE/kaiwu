import fs from 'node:fs';
import path from 'node:path';
import { api, loadSession } from '../_session.js';

function contextFromArgs(session, kwargs) {
  const domainType = kwargs['domain-type'] || session.domain_type || 'competition_stage';
  const domainId = kwargs['domain-id'] || session.stage_id;
  const experimentId = kwargs['experiment-id'] || session.experiment_id;
  const teamId = kwargs['team-id'] || session.team_id;
  const body = {
    domain: { type: domainType, id: domainId },
    experiment_id: experimentId,
  };
  if (domainType !== 'course' && teamId) body.competition_team_id = teamId;
  return body;
}

function apiPrefix(ctx) {
  return ctx.domain?.type === 'course' ? '/api/v5/Course' : '/api/v5/Competition';
}

function parseLog(raw) {
  if (typeof raw !== 'string') return { raw };
  try {
    return JSON.parse(raw);
  } catch {
    return { raw };
  }
}

async function findTask(session, kwargs, ctx, prefix) {
  if (!kwargs.name && !kwargs['task-id']) return null;
  const body = {
    ...ctx,
    page: { current: 1, size: 50 },
  };
  if (kwargs.name) body.name = kwargs.name;
  const data = await api(`${prefix}/ListTrainTask`, body, { session });
  const tasks = data.train_task || [];
  if (kwargs['task-id']) {
    const task = tasks.find(t => t.id === kwargs['task-id']);
    if (!task) throw new Error(`task-id=${kwargs['task-id']} 不在最近 50 条任务里`);
    return task;
  }
  const exact = tasks.find(t => t.name === kwargs.name);
  const task = exact || tasks[0];
  if (!task) throw new Error(`没有找到训练任务 name=${kwargs.name}`);
  return task;
}

function normalizeVar(kwargs) {
  if (kwargs.query === 'stat_log') {
    return {
      message: kwargs.message || '*',
      level: kwargs.level || '*',
      module: kwargs.module || '*',
      interval: String(kwargs.interval || 15),
    };
  }
  if (kwargs.query === 'query_log') {
    return {
      message: kwargs.message || '*',
      level: kwargs.level || '*',
      module: kwargs.module || '*',
    };
  }
  return undefined;
}

function summarize(items) {
  const countBy = (key) => {
    const out = {};
    for (const item of items) {
      const value = item[key] || '';
      out[value] = (out[value] || 0) + 1;
    }
    return out;
  };
  return {
    entries: items.length,
    levels: countBy('level'),
    modules: countBy('module'),
  };
}

export default {
  name: 'log',
  description: '拉训练任务日志。课程任务使用 Course/GetTrainLog；支持 var_level / var_module / query_log / stat_log。',
  example: 'kaiwu log --domain-type course --domain-id 2383 --experiment-id 15823 --name train-diy-v0_0 --all --output logs/train-diy-v0_0.jsonl',
  domain: 'tencentarena.com',
  args: [
    { name: 'task-id', type: 'int', required: false, help: '训练任务 id；可用 --name 代替' },
    { name: 'name', type: 'string', default: '', help: '按训练任务名查找；未给 --task-id 时使用' },
    { name: 'domain-type', type: 'string', default: '', help: 'course / competition_stage；默认使用 session' },
    { name: 'domain-id', type: 'int', default: 0, help: 'course id 或 stage id；默认使用 session.stage_id' },
    { name: 'team-id', type: 'int', default: 0, help: 'competition team id；course 日志接口不需要' },
    { name: 'experiment-id', type: 'int', default: 0, help: '实验 id；默认使用 session.experiment_id' },
    { name: 'query', type: 'string', default: 'query_log', help: 'var_level / var_module / query_log / stat_log' },
    { name: 'level', type: 'string', default: '*', help: 'query_log/stat_log 过滤：ERROR / WARNING / INFO / *' },
    { name: 'module', type: 'string', default: '*', help: 'query_log/stat_log 过滤：aisrv / env / learner / *' },
    { name: 'message', type: 'string', default: '*', help: 'query_log/stat_log message 过滤，默认 *' },
    { name: 'interval', type: 'int', default: 15, help: 'stat_log 聚合间隔秒数' },
    { name: 'limit', type: 'int', default: 20, help: 'page.size' },
    { name: 'page', type: 'int', default: 1, help: 'page.current' },
    { name: 'all', type: 'boolean', default: false, help: '分页拉完整个 query_log/stat_log，直到最后一页' },
    { name: 'start', type: 'string', default: '', help: 'ISO8601 起始时间' },
    { name: 'end', type: 'string', default: '', help: 'ISO8601 结束时间' },
    { name: 'output', type: 'string', default: '', help: '保存解析后的日志到 JSONL 文件；同时写 .summary.json' },
  ],
  columns: ['time', 'level', 'module', 'file', 'line', 'function', 'message', 'log_count', 'raw'],
  run: async (kwargs) => {
    const session = await loadSession();
    const ctx = contextFromArgs(session, kwargs);
    const prefix = apiPrefix(ctx);
    const task = await findTask(session, kwargs, ctx, prefix);
    const taskId = kwargs['task-id'] || task?.id;
    if (!taskId) throw new Error('--task-id 或 --name 至少给一个');

    const startISO = kwargs.start || task?.created_at || task?.start_time;
    const endISO = kwargs.end || task?.end_time || new Date().toISOString();
    if (!startISO) throw new Error('缺少 --start，且任务详情中没有 created_at/start_time');

    const allowed = new Set(['var_level', 'var_module', 'query_log', 'stat_log']);
    if (!allowed.has(kwargs.query)) {
      throw new Error(`未知 query=${kwargs.query}；可用: ${Array.from(allowed).join(', ')}`);
    }

    const fetchPage = async (page) => {
      const body = {
        ...ctx,
        train_task_id: taskId,
        start_time: { timestamp: startISO },
        end_time: { timestamp: endISO },
        query: kwargs.query,
        page: { size: kwargs.limit, current: page },
      };
      const varBody = normalizeVar(kwargs);
      if (varBody) body.var = varBody;
      const data = await api(`${prefix}/GetTrainLog`, body, { session });
      return data.logs || [];
    };

    const rawLogs = [];
    let page = kwargs.page;
    while (true) {
      const logs = await fetchPage(page);
      rawLogs.push(...logs);
      if (!kwargs.all || logs.length < kwargs.limit) break;
      page += 1;
      if (page > 500) throw new Error('分页超过 500 页，停止以避免无限循环');
    }

    const parsed = rawLogs.map(parseLog);
    if (kwargs.output) {
      fs.mkdirSync(path.dirname(kwargs.output), { recursive: true });
      fs.writeFileSync(kwargs.output, `${parsed.map(item => JSON.stringify(item)).join('\n')}\n`, 'utf8');
      fs.writeFileSync(`${kwargs.output}.summary.json`, JSON.stringify({
        task_id: taskId,
        task_name: task?.name || '',
        endpoint: `${prefix}/GetTrainLog`,
        query: kwargs.query,
        start: startISO,
        end: endISO,
        pages: kwargs.all ? page - kwargs.page + 1 : 1,
        ...summarize(parsed),
      }, null, 2), 'utf8');
    }

    return parsed.map(item => ({
      time: item.time || '',
      level: item.level || '',
      module: item.module || '',
      file: item.file || '',
      line: item.line || '',
      function: item.function || '',
      message: item.message || '',
      log_count: item.log_count ?? '',
      raw: item.raw ? String(item.raw) : '',
    }));
  },
};
