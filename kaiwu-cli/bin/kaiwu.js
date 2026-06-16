#!/usr/bin/env node
// kaiwu CLI 主入口：动态加载 commands/*.js + 解析 argv + 渲染输出
import fs from 'node:fs';
import path from 'node:path';
import url from 'node:url';

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));
const COMMANDS_DIR = path.join(__dirname, '..', 'commands');

async function loadCommands() {
  const files = fs.readdirSync(COMMANDS_DIR).filter(f => f.endsWith('.js') && !f.startsWith('_'));
  const cmds = {};
  for (const f of files) {
    const mod = await import(url.pathToFileURL(path.join(COMMANDS_DIR, f)).href);
    const c = mod.default;
    if (c?.name) cmds[c.name] = c;
  }
  return cmds;
}

// 解析 argv 为 kwargs。支持 --key val / --key=val / --bool / --no-bool
function parseArgs(argv, argsSpec) {
  const out = {};
  for (const a of argsSpec || []) {
    if (a.default !== undefined) out[a.name] = a.default;
  }
  const specByName = Object.fromEntries((argsSpec || []).map(a => [a.name, a]));
  for (let i = 0; i < argv.length; i++) {
    const tok = argv[i];
    if (!tok.startsWith('--')) continue;
    let key, val;
    if (tok.includes('=')) {
      [key, val] = [tok.slice(2, tok.indexOf('=')), tok.slice(tok.indexOf('=') + 1)];
    } else {
      key = tok.slice(2);
      const noPrefix = key.startsWith('no-') ? key.slice(3) : null;
      if (noPrefix && specByName[noPrefix]?.type === 'boolean') {
        out[noPrefix] = false; continue;
      }
      const next = argv[i + 1];
      const spec = specByName[key];
      if (spec?.type === 'boolean' || (!next || next.startsWith('--'))) {
        val = true;
      } else {
        val = next; i++;
      }
    }
    const spec = specByName[key];
    if (spec?.type === 'int') val = parseInt(val, 10);
    out[key] = val;
  }
  for (const a of argsSpec || []) {
    if (a.required && (out[a.name] === undefined || out[a.name] === '')) {
      throw new Error(`缺少必填参数 --${a.name}`);
    }
  }
  return out;
}

function fmtYaml(rows) {
  if (!Array.isArray(rows)) rows = [rows];
  if (!rows.length) return '(空)';
  const out = [];
  for (const r of rows) {
    if (rows.length > 1) out.push('---');
    for (const [k, v] of Object.entries(r)) {
      const s = (v === null || v === undefined) ? '~'
        : (typeof v === 'object' ? '\n  ' + JSON.stringify(v, null, 2).replace(/\n/g, '\n  ')
        : String(v));
      out.push(`${k}: ${s}`);
    }
  }
  return out.join('\n');
}

function fmtTable(rows, columns) {
  if (!Array.isArray(rows) || !rows.length) return '(空)';
  const cols = columns || Object.keys(rows[0]);
  const widths = cols.map(c => Math.max(c.length, ...rows.map(r => String(r[c] ?? '').length)));
  const sep = widths.map(w => '-'.repeat(w)).join('-+-');
  const head = cols.map((c, i) => c.padEnd(widths[i])).join(' | ');
  const body = rows.map(r => cols.map((c, i) => String(r[c] ?? '').padEnd(widths[i])).join(' | '));
  return [head, sep, ...body].join('\n');
}

function fmtJson(rows) {
  return JSON.stringify(rows, null, 2);
}

function help(cmds, name) {
  if (name && cmds[name]) {
    const c = cmds[name];
    console.log(`\n  kaiwu ${c.name}\n  ${c.description || ''}\n`);
    if (c.example) console.log(`  示例: ${c.example}\n`);
    if (c.args?.length) {
      console.log('  参数:');
      for (const a of c.args) {
        const req = a.required ? '*' : ' ';
        const def = a.default !== undefined ? ` (默认 ${JSON.stringify(a.default)})` : '';
        console.log(`    ${req} --${a.name.padEnd(22)} [${a.type || 'string'}] ${a.help || ''}${def}`);
      }
    }
    return;
  }
  console.log(`
  kaiwu — 腾讯开悟比赛工作台 CLI

  用法:
    kaiwu <command> [--args]
    kaiwu <command> --help    查看子命令参数
    kaiwu --format json|yaml|table  指定输出格式（默认 yaml）

  命令分类:
    会话      login / logout / whoami
    元信息    experiment / team-info / list-versions / ide-status
    资源      resource-balance / resource-stat
    训练任务  list-train-task / get-train-task / create-train-task / release-train-task
    训练产物  list-models
    模型库    list-ai-models / submit-model
    监控      metric / log / open-monitor
    评估对战  list-battle-tasks / get-battle-task / list-games / create-battle-task / stop-battle-task
    比赛      list-races

  全部命令:`);
  const names = Object.keys(cmds).sort();
  const W = Math.max(...names.map(n => n.length));
  for (const n of names) {
    console.log(`    ${n.padEnd(W + 2)}${cmds[n].description || ''}`);
  }
  console.log();
}

async function main() {
  const argv = process.argv.slice(2);
  const cmds = await loadCommands();

  // 全局 flag
  let format = 'yaml';
  const fIdx = argv.indexOf('--format');
  if (fIdx >= 0) { format = argv[fIdx + 1]; argv.splice(fIdx, 2); }
  const fShortIdx = argv.indexOf('-f');
  if (fShortIdx >= 0) { format = argv[fShortIdx + 1]; argv.splice(fShortIdx, 2); }

  if (!argv.length || argv[0] === 'help' || argv[0] === '--help' || argv[0] === '-h') {
    help(cmds, argv[1]); return;
  }

  const name = argv[0];
  const cmd = cmds[name];
  if (!cmd) {
    console.error(`未知命令: ${name}`);
    console.error(`运行 'kaiwu help' 查看全部命令`);
    process.exit(1);
  }

  if (argv.includes('--help') || argv.includes('-h')) {
    help(cmds, name); return;
  }

  let kwargs;
  try {
    kwargs = parseArgs(argv.slice(1), cmd.args || []);
  } catch (e) {
    console.error(e.message);
    help(cmds, name);
    process.exit(1);
  }

  try {
    const result = await cmd.run(kwargs);
    if (result === undefined || result === null) return;
    const out = format === 'json' ? fmtJson(result)
              : format === 'table' ? fmtTable(result, cmd.columns)
              : fmtYaml(result);
    console.log(out);
  } catch (e) {
    console.error(`error: ${e.message}`);
    if (process.env.KAIWU_DEBUG) console.error(e.stack);
    process.exit(1);
  }
}

main();
