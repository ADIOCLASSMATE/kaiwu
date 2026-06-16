import { loadSession } from '../_session.js';
import { authDownloadAndSave } from '../_download_util.js';

const DEFAULT_KEY = 'public/abs_tools/ABSParsingTool_hok_g_shelled_v1.0.2.zip';

export default {
  name: 'download-abs-tool',
  description: '下载开悟官方提供的 ABS 录像播放器工具 zip',
  example: 'kaiwu download-abs-tool --output ./ABSParsingTool.zip',
  domain: 'tencentarena.com',
  args: [
    { name: 'file-key', type: 'string', default: DEFAULT_KEY, help: '工具 zip 的 cos file_key（默认 v1.0.2）' },
    { name: 'output', type: 'string', default: '', help: '本地保存路径，缺省取服务端 filename' },
  ],
  columns: ['file_key', 'size_mb', 'output'],
  run: async (kwargs) => {
    const session = await loadSession();
    const fileKey = kwargs['file-key'];
    const out = kwargs.output || fileKey.split('/').pop() || 'abs_tool.zip';
    const r = await authDownloadAndSave(session, fileKey,
      { type: 'common', id: 0 }, out);
    return [{ file_key: fileKey, size_mb: r.size_mb, output: r.output }];
  },
};
