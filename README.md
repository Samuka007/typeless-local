# typeless-local — 本地语音输入 + 在线 LLM 润色自部署方案

对标 [Typeless](https://www.typeless.com/) / Wispr Flow 的听写体验：**按住热键说话 → 松开 → 润色好的文字自动打进光标处**。语音识别完全本地（AMD GPU 加速），文本润色走在线 LLM API（可选）。

## 一、硬件适配结论（本机：RX 6750 GRE 12GB / 5800X3D / 32GB / Win11）

| 路线 | 结论 |
|---|---|
| ROCm (PyTorch / whisper.cpp ROCm 版) | ❌ 官方仅支持 gfx110X（RX 7000+），6750 GRE 是 **gfx1031**，不在支持列表 |
| ONNX Runtime + DirectML | ⚠️ 可用但算子覆盖一般，whisper 生态支持弱 |
| **whisper.cpp + Vulkan（自编译）** | ✅ **首选，已实测跑通**。注意：官方 `whisper-bin-x64.zip` 是 **纯 CPU 构建（AVX2）**，GPU 加速需按 `scripts/build-vulkan.ps1` 本地编译（MinGW 工具链 + LunarG Vulkan SDK，脚本已固化全流程） |

## 二、架构

```
麦克风 ──热键录音(16kHz wav)──▶ FastAPI(127.0.0.1:8765)
                                   │ /dictate
                                   ├─▶ whisper.cpp server (127.0.0.1:8178, Vulkan GPU)
                                   │     ggml-small (~0.5s) 或 large-v3-turbo (~1.5s)
                                   └─▶ LLM API (DeepSeek/Qwen/GLM..., OpenAI 兼容)
                                         润色/纠错/标点，失败自动降级为原文
                                   ▼
                        模拟 Ctrl+V 粘贴到光标处（或模拟键入）
```

- **STT**：[ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) 官方 `whisper-bin-x64.zip`（含 Vulkan 版 ggml 后端 + 服务端程序）
- **备选 STT**：[lemonade-sdk/whisper.cpp-amd](https://github.com/lemonade-sdk/whisper.cpp-amd)（AMD 官方维护，ROCm 版需 RX 7000+；本项目发布后可直接替换下载源）
- **备选流式方案**：[k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)（zipformer 流式中英模型，纯 CPU 实时，首字延迟最低，适合边说边出字的模式）
- **成品参考**：[TypeWhisper/typewhisper-win](https://github.com/TypeWhisper/typewhisper-win)（C#/WPF 整合型，含翻译）、[cjcscssvant/openwhispr](https://github.com/cjcscssvant/openwhispr)、[cjpais/Handy](https://github.com/cjpais/Handy)（Tauri 跨平台）

## 三、安装

```powershell
cd typeless-local
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
```

脚本自动完成：下载官方 whisper.cpp Windows 构建（纯 CPU，用于回退）→ 下载 `ggml-small.bin` / `ggml-large-v3-turbo.bin` / silero VAD 模型 → 创建 `.venv` 并安装依赖。

**GPU 加速（必做一步）**——官方预编译包没有 Vulkan 后端，需本地编译一次（需 `scoop install vulkan`，其余工具链脚本会自动探测）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build-vulkan.ps1
```

产物在 `build/vulkan/bin/whisper-server.exe`，`.env` 默认已指向它。

然后：

```powershell
copy .env.example .env     # 填入 LLM_API_KEY（DeepSeek/Qwen/GLM/SiliconFlow 均可）
uv run typeless-server     # 启动 ASR 服务（自动拉起 whisper-server）
```

Python 环境由 **uv** 管理：`uv sync` 按 `uv.lock` 精确复现依赖（秒级），`.python-version` 固定 3.13。没有 uv 时 setup.ps1 会自动安装。

## 四、使用

```powershell
# 另开一个终端
uv run typeless-dictate               # Alt 单击开始、再击结束(Typeless 式);设置存 settings.json
uv run typeless-dictate --key f9      # CLI 覆盖热键(优先级最高)
uv run typeless-dictate --trigger hold # 改为按住说话
uv run typeless-dictate --mode type   # 模拟逐字键入(替代剪贴板粘贴)
uv run typeless-dictate --raw         # 跳过 LLM 润色,只出原始转写
```

### WebUI 控制台(推荐)

浏览器打开 **http://127.0.0.1:8765/ui**:

- **状态卡**:ASR/LLM 健康灯 + 实时 ● 录音指示(客户端通过 /client-status 上报)
- **交互设置**:热键、触发方式(单击切换 / 按住说话)、输出方式(粘贴 / 键入)、悬浮 REC 指示器 —— 保存后**客户端即时生效**(settings.json 热加载,无需重启)
- **AI 润色**:开关 + 风格(默认 / 邮件 / 聊天 / 笔记)+ **自定义提示词**(附加给 LLM 的额外规则,优先级高于内置风格,保存即时生效)
- **LLM API**:Base URL / 模型名 / API Key 全部页内可改,带**测试连接**按钮;保存后 LLM 客户端热重建,无需重启服务。另有**输出预算(max_tokens)**与**思考模式**两个进阶旋钮:推理模型(deepseek-flash 等)会先把预算花在思考链上,预算太小会让长句只思考不出文(表现为无标点原文直出);思考模式默认关闭(润色最快),设为「模型默认」可让模型先思考再润色
- **Whisper 引擎**:模型下拉(自动列出 models/ 下全部)、语言(auto/zh/en)、whisper 提示词;换模型后点**应用新模型**重启引擎(缓存热时约 3~30s)
- **最近听写**:最近 100 条(展示 50)历史,点击即复制润色结果

配置优先级:**settings.json 是唯一真相源**。首次运行时从 `.env` 播种,旧版 settings.json 会自动迁移缺失的 LLM/whisper 字段;之后 WebUI 改动全部写入 settings.json(`.env` 不再生效,删除 settings.json 即可回退到 .env)。端口等真正的部署常量仍留在 `.env`。

录音期间按 `Esc` 立即取消本次。剪贴板内容在粘贴后 1 秒自动恢复。

### Windows 热键实现（最佳实践说明）

- **钩子选型**：用 `pynput`（底层 `WH_KEYBOARD_LL` 钩子），弃用 `keyboard` 库——后者在钩子线程里忙轮询，回调慢会拖慢**全系统**的按键响应；pynput 的钩子回调只做入队，重活全部在 asyncio 任务里执行，钩子路径亚毫秒。
- **按键回显**：pynput 全局抑制是全有或全无，无法只吞掉单个键，所以热键选**不产生字符的键**（F 键区 / scroll_lock / pause），避免吞掉正常打字。
- **管理员窗口**：Windows UIPI 隔离——普通权限进程的模拟输入进不了管理员权限的窗口（终端/任务管理器）。需要在那些窗口听写时，以管理员身份运行 `typeless-dictate`。
- **输出方式**：默认剪贴板粘贴（快，长文本也瞬时）；`--mode type` 用 `KEYEVENTF_UNICODE` 逐字注入（CJK 安全，但长文本慢）。
- **开机自启**：用任务计划程序注册登录时启动，比注册表 Run 键更可控（可加最高权限、延迟启动）：`schtasks /create /tn typeless /sc onlogon /tr "..."`

## 五、实测性能（RX 6750 GRE @ Vulkan，6.1s 测试音频，稳态）

| 环节 | 延迟 | 说明 |
|---|---|---|
| large-v3-turbo encode（GPU 稳态） | ~0.3–1.3s | 首请求含 shader 编译会慢数倍，之后稳定 |
| whisper-server 推理（`language=auto`） | ~3.0s | auto 要先跑一次完整 encode 做语言检测 |
| whisper-server 推理（`language=en/zh` 固定） | **~1.7s** | 跳过语言检测，快 43% |
| 中文 steering prompt 开销 | +0.5s/请求 | 默认已关闭（`WHISPER_PROMPT=`） |
| FastAPI 转发开销 | ~0s | 已共享 httpx 连接池（trust_env=False） |
| LLM 润色（DeepSeek 级） | ~1–3s | 失败自动降级为原文，不丢字 |
| **端到端（松开出字）** | **~3s（auto）/ ~2s（固定语言）+ LLM** | |

对照：同模型纯 CPU（16 线程）encode 需 ~4.4s，GPU 提速 **~3.4 倍**。

进一步提速的选项：
- `.env` 里 `WHISPER_LANGUAGE=zh`（或 `en`）→ 直接省掉 1.3s，代价是中英混说时另一语言识别变差
- `WHISPER_MODEL=models/ggml-small.bin` → CPU 都能实时，GPU 上 ~0.3s；追求极速换 `ggml-base.bin`
- LLM 换低延迟供应商（Qwen-turbo / SiliconFlow），或改 `rephrase.py` 为流式 + 边收边打
- 纯本地 LLM（12GB 显存可跑 Qwen3-4B/8B 量化，用 llama.cpp Vulkan 版），完全离线但延迟更高

## 六、进阶方向

1. **流式听写**：换 sherpa-onnx 流式 zipformer，边说边出字，热键只管开始/结束
2. **斜杠命令**：在 dictation 里说 "new line" / "中文模式" 等指令映射为动作
3. **自定义词典**：把个人术语加进 whisper `prompt` 与 LLM system prompt，提高专有名词准确率
4. **托盘常驻**：pystray 包装成开机自启的系统托盘应用
5. **多 profile**：邮件风格 / 代码注释 / 聊天风格，不同 system prompt 一键切换
6. **本地 LLM 路线**：llama.cpp Vulkan 版跑 Qwen3-8B-Q4，实现 100% 离线（延迟 +2-4s）

## 七、故障排查

### 生命周期契约（重要，一晚调试换来的教训）

- **启动顺序**：先 `uv run typeless-server`（终端 1），再 `uv run typeless-dictate`（终端 2）。dictate 启动会预检并明确提示，不会再甩 404。
- **whisper-server 是独立常驻服务**：模型加载一次后保温显存；重启 FastAPI/dictate **不会**也不需要重启它。手动全停：`taskkill /F /IM whisper-server.exe`。
- **单实例是生死线**：whisper.cpp 用 SO_REUSEADDR 绑端口，多个实例会同时 LISTEN，连接被随机路由到健康或僵死实例——"时好时坏"的玄学几乎都是它。`ensure_server` 会在拉起前清理并验证清零，但别自己开第二个。
- **端口只有 `WHISPER_PORT` 一个真相源**：URL 由它派生。若 `.env` 里还有单独的 `WHISPER_URL` 且与端口不一致，会出现「curl 打 8178 全通、Python 打 8179 全拒（10061）」的完美分裂——这不是玄学，是配置。

- **`whisper-server did not come up`**：确认显卡驱动已装（Vulkan 运行时随 Adrenalin 驱动提供）；运行 `bin\whisper-cli.exe --version` 看是否报缺 DLL
- **转写全是英文**：`.env` 设 `WHISPER_LANGUAGE=zh`，或在 `WHISPER_PROMPT=` 填 `以下是普通话的句子。`（注意 prompt 有 ~0.5s/请求开销）
- **中英混说识别差**：保持 `WHISPER_LANGUAGE=auto`（多花 1.3s 语言检测，但混说最稳）
- **`keyboard` 库无效**：Windows 下需以普通权限运行；某些安全软件会拦截全局键盘钩子
- **LLM 超时**：`rephrase.py` 默认 6s 超时自动降级为原文，可在 `server/rephrase.py` 调整
