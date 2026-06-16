import readline from 'node:readline';
import { rawSignIn, saveSession, saveCredentials, loadCredentials, getSessionPath, getCredentialsPath } from '../_session.js';

function jwtPayload(tk) {
  try {
    const part = tk.split('.')[1].replace(/-/g, '+').replace(/_/g, '/');
    return JSON.parse(Buffer.from(part, 'base64').toString());
  } catch { return null; }
}

// readline prompt（密码输入静默）
function prompt(q, { silent = false } = {}) {
  return new Promise(resolve => {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout, terminal: true });
    if (silent) {
      const stdin = process.stdin;
      const onData = (ch) => {
        ch = ch.toString();
        if (ch === '\n' || ch === '\r' || ch === '') return;
        process.stdout.write('*');
      };
      stdin.on('data', onData);
      rl.question(q, ans => { stdin.off('data', onData); process.stdout.write('\n'); rl.close(); resolve(ans); });
    } else {
      rl.question(q, ans => { rl.close(); resolve(ans.trim()); });
    }
  });
}

async function chooseLoginMethod() {
  console.log('\n  请选择登录方式:');
  console.log('    1) 手机号 + 密码');
  console.log('    2) 邮箱 + 密码');
  while (true) {
    const ans = await prompt('  输入 1 / 2: ');
    if (ans === '1') return 'phone';
    if (ans === '2') return 'email';
    console.log('  无效选择，请输入 1 或 2');
  }
}

export default {
  name: 'login',
  description: '登录开悟（持久化凭据，token 过期自动刷新）。第一次交互式输入手机号/邮箱+密码',
  example: 'kaiwu login   # 交互式 / kaiwu login --phone 15084735093 --password xxx',
  args: [
    { name: 'phone', type: 'string', help: '中国大陆手机号（自动加 +86-）' },
    { name: 'email', type: 'string', help: '邮箱（与 --phone 二选一）' },
    { name: 'password', type: 'string', help: '密码 12-20 字符（缺省读 KAIWU_PASSWORD env 或交互输入）' },
    { name: 'stage-id', type: 'int', default: 446, help: '比赛阶段 id' },
    { name: 'team-id', type: 'int', default: 8365, help: '队伍 id' },
    { name: 'experiment-id', type: 'int', default: 11427, help: '实验 id' },
    { name: 'no-save-password', type: 'boolean', default: false, help: '不持久化密码（token 过期需重新 login）' },
  ],
  columns: ['key', 'value'],
  run: async (kwargs) => {
    let { phone, email, password } = kwargs;

    // 交互模式：缺关键参数时进入
    if (!phone && !email && !process.env.KAIWU_PHONE && !process.env.KAIWU_EMAIL) {
      const method = await chooseLoginMethod();
      if (method === 'phone') {
        phone = await prompt('  手机号（如 15084735093）: ');
      } else {
        email = await prompt('  邮箱: ');
      }
    } else {
      phone = phone || process.env.KAIWU_PHONE;
      email = email || process.env.KAIWU_EMAIL;
    }

    if (!password) password = process.env.KAIWU_PASSWORD;
    if (!password) password = await prompt('  密码（12-20 字符）: ', { silent: true });

    if (password.length < 12 || password.length > 20) {
      throw new Error(`密码必须 12-20 字符，当前 ${password.length}`);
    }

    let creds;
    if (phone) {
      const norm = phone.startsWith('+') ? phone : `+86-${phone.replace(/^\+?86-?/, '')}`;
      creds = { phone: norm, password };
    } else {
      creds = { email, password };
    }

    const data = await rawSignIn(creds);
    const payload = jwtPayload(data.token);
    const session = {
      token: data.token,
      expire_at: data.expire,
      user_id: payload?.custom,
      domain_type: 'competition_stage',
      stage_id: kwargs['stage-id'],
      team_id: kwargs['team-id'],
      experiment_id: kwargs['experiment-id'],
    };
    saveSession(session);
    if (!kwargs['no-save-password']) saveCredentials(creds);

    return [
      { key: 'result', value: 'login OK' },
      { key: 'user_id', value: session.user_id },
      { key: 'token', value: data.token.slice(0, 24) + '...' + data.token.slice(-8) },
      { key: 'expire_at', value: data.expire },
      { key: 'password_status', value: data.password_status },
      { key: 'phone_status', value: data.phone_status },
      { key: 'session_file', value: getSessionPath() },
      { key: 'credentials_file', value: kwargs['no-save-password'] ? '(未持久化)' : getCredentialsPath() },
      { key: 'auto_refresh', value: kwargs['no-save-password'] ? '关闭' : '开启（token 过期自动用持久化密码续期）' },
    ];
  },
};
