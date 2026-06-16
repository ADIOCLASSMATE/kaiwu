import { loadSession, hasCredentials, loadCredentials, getSessionPath, getCredentialsPath } from '../_session.js';

export default {
  name: 'whoami',
  description: '看当前登录状态：user_id / token 剩余有效期 / 比赛上下文 / 是否启用 auto-refresh',
  example: 'kaiwu whoami',
  args: [],
  columns: ['key', 'value'],
  run: async () => {
    const session = await loadSession();
    const remainMs = new Date(session.expire_at).getTime() - Date.now();
    const remainDays = (remainMs / 86400000).toFixed(1);
    const creds = hasCredentials() ? loadCredentials() : null;
    return [
      { key: 'user_id', value: session.user_id },
      { key: 'logged_in_as', value: creds?.phone || creds?.email || '(凭据未持久化)' },
      { key: 'stage_id', value: session.stage_id },
      { key: 'team_id', value: session.team_id },
      { key: 'experiment_id', value: session.experiment_id },
      { key: 'token_expire_at', value: session.expire_at },
      { key: 'token_remaining', value: `${remainDays} 天` },
      { key: 'auto_refresh', value: hasCredentials() ? '开启' : '关闭（无持久化凭据）' },
      { key: 'session_file', value: getSessionPath() },
      { key: 'credentials_file', value: hasCredentials() ? getCredentialsPath() : '(无)' },
    ];
  },
};
