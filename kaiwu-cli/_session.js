// kaiwu session: 凭据持久化 + token 自动刷新 + 签名 + base fetch
// ~/.kaiwu/credentials.json  长期：phone/email + password（chmod 600）
// ~/.kaiwu/session.json      短期：token + 比赛上下文（自动 refresh）
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const SESSION_DIR = path.join(os.homedir(), '.kaiwu');
const SESSION_PATH = path.join(SESSION_DIR, 'session.json');
const CREDENTIALS_PATH = path.join(SESSION_DIR, 'credentials.json');
const BASE = 'https://tencentarena.com';

export function getSessionPath() { return SESSION_PATH; }
export function getCredentialsPath() { return CREDENTIALS_PATH; }
export function hasSession() { return fs.existsSync(SESSION_PATH); }
export function hasCredentials() { return fs.existsSync(CREDENTIALS_PATH); }

function readJson(p) { return JSON.parse(fs.readFileSync(p, 'utf8')); }
function writeJson(p, obj, mode = 0o600) {
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, JSON.stringify(obj, null, 2));
  fs.chmodSync(p, mode);
}

export function loadSessionRaw() {
  if (!fs.existsSync(SESSION_PATH)) return null;
  return readJson(SESSION_PATH);
}

export function saveSession(s) { writeJson(SESSION_PATH, s); }
export function clearSession() { if (fs.existsSync(SESSION_PATH)) fs.unlinkSync(SESSION_PATH); }

export function loadCredentials() {
  if (!fs.existsSync(CREDENTIALS_PATH)) return null;
  return readJson(CREDENTIALS_PATH);
}
export function saveCredentials(c) { writeJson(CREDENTIALS_PATH, c); }
export function clearCredentials() { if (fs.existsSync(CREDENTIALS_PATH)) fs.unlinkSync(CREDENTIALS_PATH); }

// 签名: hash(ts + token.slice(-32) + path末段) & 0x7FFFFFFF (DJB2 变种)
// 来自开悟 webpack module 73916
export function sign(ts, token, urlPath) {
  const last = urlPath.split('/').slice(-1)[0] || '';
  const s = String(ts) + (token || '').slice(-32) + last;
  let t = 5381;
  for (let i = 0; i < s.length; i++) {
    t = (t + (((t << 5) | 0) + s.charCodeAt(i))) | 0;
  }
  return 0x7FFFFFFF & t;
}

function jwtPayload(tk) {
  try {
    const part = tk.split('.')[1].replace(/-/g, '+').replace(/_/g, '/');
    return JSON.parse(Buffer.from(part, 'base64').toString());
  } catch { return null; }
}

// 直接调 SignIn 不依赖现有 session（用于登录 / token refresh）
async function rawSignIn(creds) {
  const body = creds.email
    ? { address: { address: creds.email, type: 'email' }, email: creds.email, password: creds.password }
    : { address: { address: creds.phone, type: 'phone' }, phone: creds.phone, password: creds.password };
  const ts = Math.floor(Date.now() / 1000);
  const auth = sign(ts, '', '/api/v5/User/SignIn');
  const res = await fetch(BASE + '/api/v5/User/SignIn', {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      'Accept-Language': 'zh',
      'Content-Type': 'application/json',
      'x-kaiwu-ts': String(ts),
      'x-kaiwu-auth': String(auth),
    },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (data.code !== 0) {
    const err = new Error(`SignIn 失败: ${data.code} ${data.msg || data.reason}`);
    err.code = data.code;
    throw err;
  }
  return data.data; // { token, expire, password_status, phone_status, ... }
}

// 用现有 credentials 刷新 token，写回 session.json
export async function refreshSession() {
  const creds = loadCredentials();
  if (!creds) throw new Error('没有持久化凭据。请运行: kaiwu login');
  const oldSession = loadSessionRaw() || {};
  const data = await rawSignIn(creds);
  const payload = jwtPayload(data.token);
  const session = {
    token: data.token,
    expire_at: data.expire,
    user_id: payload?.custom ?? oldSession.user_id,
    domain_type: oldSession.domain_type || 'competition_stage',
    stage_id: oldSession.stage_id,
    team_id: oldSession.team_id,
    experiment_id: oldSession.experiment_id,
  };
  saveSession(session);
  return session;
}

// 加载 session：token 过期或不存在时自动 refresh（如有 credentials）
export async function loadSession({ autoRefresh = true } = {}) {
  let s = loadSessionRaw();
  const expired = s && s.expire_at && new Date(s.expire_at) < new Date();
  if (!s || expired) {
    if (!autoRefresh) {
      if (!s) throw new Error('session 未初始化。请运行: kaiwu login');
      throw new Error(`token 已过期 (${s.expire_at})。重新运行: kaiwu login`);
    }
    if (!hasCredentials()) {
      throw new Error('session 未初始化且无持久化凭据。请运行: kaiwu login');
    }
    s = await refreshSession();
  }
  return s;
}

// 同步版本（不 refresh）—— 用于 login 等不需要现有 token 的场景
export function loadSessionSync() {
  const s = loadSessionRaw();
  if (!s) throw new Error('session 未初始化。请运行: kaiwu login');
  return s;
}

// 调用开悟 API。urlPath 形如 "/api/v5/Competition/ListTrainTask"
// opts.anonymous=true → SignIn 等不需 token 的 endpoint
// opts.session = 显式传 session 对象（避免 reload）
// 自动 token-refresh: 1040 错误时用 credentials 刷新一次重试
export async function api(urlPath, body = {}, opts = {}) {
  const exec = async (session) => {
    const ts = Math.floor(Date.now() / 1000);
    const token = session?.token || '';
    const auth = sign(ts, token, urlPath);
    const headers = {
      'Accept': 'application/json',
      'Accept-Language': 'zh',
      'Content-Type': 'application/json',
      'x-kaiwu-ts': String(ts),
      'x-kaiwu-auth': String(auth),
      ...(opts.headers || {}),
    };
    if (token) headers['Authorization'] = `Bearer ${token}`;
    const res = await fetch(BASE + urlPath, {
      method: opts.method || 'POST',
      headers,
      body: opts.method === 'GET' ? undefined : JSON.stringify(body),
    });
    const text = await res.text();
    let data;
    try { data = JSON.parse(text); } catch { data = { raw: text }; }
    return data;
  };

  let session = opts.session ?? (opts.anonymous ? null : await loadSession());
  let data = await exec(session);

  // token 校验失败 → 自动 refresh 重试一次
  if (data.code === 1040 && !opts.anonymous && hasCredentials() && !opts._retried) {
    session = await refreshSession();
    data = await exec(session);
  }

  if (data.code !== 0 && data.code !== undefined) {
    const err = new Error(`kaiwu api error ${data.code}: ${data.msg || data.reason || 'unknown'}`);
    err.code = data.code;
    err.response = data;
    throw err;
  }
  return data.data ?? data;
}

// 注入比赛上下文到 body
export function withCtx(session, extra = {}) {
  return {
    domain: { type: session.domain_type || 'competition_stage', id: session.stage_id },
    experiment_id: session.experiment_id,
    competition_team_id: session.team_id,
    ...extra,
  };
}

// 从命令行参数构建请求上下文（body.domain / experiment_id / competition_team_id）
export function contextFromArgs(session, kwargs) {
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

// 根据上下文返回 API 前缀：course 域用 /api/v5/Course，否则用 /api/v5/Competition
export function apiPrefix(ctx) {
  return ctx.domain?.type === 'course' ? '/api/v5/Course' : '/api/v5/Competition';
}

// 通用 domain/experiment 参数定义，供各命令 args 复用
export const DOMAIN_ARGS = [
  { name: 'domain-type', type: 'string', default: '', help: 'course / competition_stage；默认使用 session' },
  { name: 'domain-id', type: 'int', default: 0, help: 'course id 或 stage id；默认使用 session.stage_id' },
  { name: 'experiment-id', type: 'int', default: 0, help: '实验 id；默认使用 session.experiment_id' },
];

export { rawSignIn };
export const BASE_URL = BASE;
