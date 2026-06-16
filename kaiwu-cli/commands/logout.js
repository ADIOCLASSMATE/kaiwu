import { clearSession, clearCredentials, hasSession, hasCredentials, getSessionPath, getCredentialsPath } from '../_session.js';

export default {
  name: 'logout',
  description: '清除本地 session + 凭据。下次需要重新 kaiwu login',
  example: 'kaiwu logout',
  args: [
    { name: 'keep-credentials', type: 'boolean', default: false, help: '只清 session.json 保留密码（再次 kaiwu list-* 会自动重登）' },
  ],
  columns: ['key', 'value'],
  run: async (kwargs) => {
    const sBefore = hasSession();
    const cBefore = hasCredentials();
    clearSession();
    if (!kwargs['keep-credentials']) clearCredentials();
    return [
      { key: 'session_cleared', value: sBefore ? getSessionPath() : '(本来就没有)' },
      { key: 'credentials_cleared', value: kwargs['keep-credentials'] ? '(保留)' : (cBefore ? getCredentialsPath() : '(本来就没有)') },
    ];
  },
};
