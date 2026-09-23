# WeChat Agent

一个基于 LangGraph 的微信聊天替身原型。用户提供聊天记录（文本、JSON、CSV、PDF、截图等），系统自动解析并构建好友/群聊画像，再结合短期会话记忆生成拟人回复，并在发送前经过安全校验。

默认使用 `MockGateway` 进行本地模拟；可选开启 `VisionGateway`，通过截图监控本地 macOS 微信聊天窗口并自动回复（Windows 为占位实现）。

## 架构

按设计文档划分为六层：

- **L1 接入层**：`MockGateway`（本地模拟收发） / `VisionGateway`（macOS 微信窗口视觉监控）
- **L2 编排层**：LangGraph StateGraph
- **L3 Agent 层**：Auto Reply、Session Manager、Safety Guard、Profile Builder
- **L4 工具层**：画像/会话检索工具
- **L5 记忆层**：JSON 本地画像 + 内存/Redis 短期会话
- **L6 校验层**：敏感词拦截 + LLM 自审
- **数据摄入层**：`wechat_agent.ingestion` — 解析用户提供的聊天记录文件

## 快速开始

### 1. 安装依赖

```bash
pip install -e ".[dev]"
# 如果需要解析 PDF 或图片聊天记录，额外安装：
pip install -e ".[ingestion]"
# 如果需要视觉监控真实微信窗口（macOS），额外安装：
pip install -e ".[vision]"   # 只需 Pillow + pyobjc-framework-Quartz
cp .env.example .env
```

编辑 `.env`，填入大模型配置：

```bash
# 例如火山方舟 OpenAI-compatible：
LLM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
LLM_API_KEY=你的 API Key
LLM_MODEL=ep-xxxxxxxxxx
LLM_API_TYPE=openai  # openai | anthropic
```

视觉监控需要模型支持 vision（多模态）；若当前模型无法识别截图或生成空回复，请更换为对应的 vision 模型。

### 2. 导入聊天记录并生成画像

支持格式：

| 格式 | 说明 |
|---|---|
| `.txt` | 每行一条消息，格式：`发送者: 内容` |
| `.json` / `.jsonl` | 消息对象数组 |
| `.csv` | 必需列：`sender_id`, `sender_name`, `content` |
| `.pdf` | 导出聊天记录 PDF，由 LLM 自动结构化（需 `[ingestion]`） |
| `.png/.jpg/.jpeg` | 聊天截图，直接传给多模态 LLM 识别，会自动提取顶部聊天标题作为 `friend_name` |

```bash
python scripts/import_chat_history.py friend_001 data/sample_chat.json
```

PDF 或截图示例：

```bash
python scripts/import_chat_history.py friend_001 ~/Downloads/chat_export.pdf
python scripts/import_chat_history.py friend_001 ~/Downloads/wechat_screenshot.png
```

> 截图导入时，LLM 会同时识别截图顶部的聊天标题并写入画像的 `friend_name`。如果识别结果有问题（比如乱码或和微信窗口标题不一致），可以手动覆盖：
>
> ```bash
> python scripts/import_chat_history.py friend_001 ~/Downloads/wechat_screenshot.png --friend-name "真实昵称"
> ```

### 3. 模拟单轮对话

```bash
wechat-agent simulate \
  --chat-id friend_001 \
  --message "周末有空吃饭吗？"
```

### 4. Mock 端到端测试

```bash
python scripts/test_send_wechat.py
```

会依次注入模拟消息，验证画像加载、会话记忆、回复生成、安全校验和模拟发送全流程。

### 5. 视觉实时监控真实微信窗口（macOS）

需要先安装视觉依赖：

```bash
pip install -e ".[vision]"
```

并确保终端已获得 **屏幕录制** 和 **辅助功能/辅助控制** 权限：

- 系统设置 → 隐私与安全性 → **屏幕录制**：勾选你的终端（Terminal / iTerm）和 Python。
- 系统设置 → 隐私与安全性 → **辅助功能**：勾选你的终端（Terminal / iTerm）和 Python。

如果列表里没有 Python，可以点 **+**，把 `which python` 返回的路径加进去。

也可以用更方便的 bash 脚本启动：

```bash
# 默认 dry-run，安全先测
./scripts/run_vision.sh --chat-id "friend_001"

# 确认无误后再开启真实发送
./scripts/run_vision.sh --chat-id "friend_001" --send --ticks 0
```

如果你连导入聊天记录也不想手动分步执行，可以直接用完整端到端脚本：

```bash
# 导入聊天记录 + 生成画像 + 启动视觉监控（dry-run）
./scripts/run_e2e.sh \
  --chat-id "friend_001" \
  --chat-file ~/Downloads/chat_history.json
   --ticks 0 # 用于实时监控

# 同一套数据，确认无误后开启真实发送 提供最新的聊天记录截图，然后运行此命令可以实时监测聊天窗口运行，以这个指令为准
./scripts/run_e2e.sh \
  --chat-id "friend_001" \
  --chat-file ~/Downloads/chat_history.json # 或者提供最新的聊天记录截图 \
  --send
  --ticks 0 # 用于实时监控
```

强烈推荐使用以下指令实现后台实时监控
```bash

# 如果希望针对一个新的聊天窗口开启自动聊天回复，提供最新的聊天记录截图，运行该指令即可。chat-id要做明显区分，poll-interval 调整可以大一点，防止限流
./scripts/run_e2e.sh --chat-id "friend_002" --send --ticks 0 --chat-file data/截屏2026-09-23.png --poll-interval 60
```

#### 5.1 Dry-run 测试（只识别不发送）

```bash
python scripts/test_vision_friend.py \
  --chat-id "friend_001" \
  --dry-run \
  --ticks 0
```

#### 5.2 真实自动发送

```bash
python scripts/run_vision_friend.py \
  --chat-id "friend_001" \
  --confirm-send \
  --ticks 0
```

- `--chat-id`：`data/profiles/` 中对应画像文件名。脚本会自动从该画像里读取 `friend_name`，用来在截图里匹配具体的聊天标题。
- `--window-title`：显式指定**窗口标题正则**，用于 macOS 上定位微信窗口；默认是 `WeChat|微信`，一般不需要再传。

> 说明：`./scripts/run_vision.sh` 和 `./scripts/run_e2e.sh` 不再接受 `--friend-name` 参数。它们会自动从画像里读取并内部传给底层 Python 脚本，用来在截图中匹配聊天标题。macOS 上微信窗口的实际标题通常是 `微信`，所以默认用 `WeChat|微信` 定位窗口；只有窗口标题确实不一样时，才需要用 `--window-title` 覆盖。
- `--poll-interval`：截图轮询间隔，默认 30.0 秒。
- `--ticks`：跑多少轮后自动停止，默认 3 轮；传 `0` 则一直轮询，按 `Ctrl+C` 停止。

#### 5.3 工作机制与调优

- **只在检测到新消息时才回复**：每轮截图会先对比画面哈希（dHash）。如果画面变化很小（汉明距离 ≤ `VISION_CHANGE_THRESHOLD`），就跳过本轮，不调用 LLM。
- **同一条消息不会重复回复**：当画面变化后，VisionGateway 会提取出最新的一条对方消息，并生成 digest。如果这条消息和上一轮处理过的 digest 相同，就说明已经回复过了，直接跳过。
- **发送回复不会覆盖已见消息 digest**：`_last_seen_digest` 专门记录已处理的好友消息，`_last_sent_digest` 记录自己发出去的回复，二者分开，避免“刚发完回复又把原消息当新消息”的循环。

相关环境变量（可在 `.env` 中调整）：

```ini
VISION_POLL_INTERVAL=30.0         # 轮询间隔
VISION_CHANGE_THRESHOLD=15        # 画面变化阈值，越小越敏感
VISION_WINDOW_WAIT_TIMEOUT=30.0   # 启动后等待微信窗口出现的最大秒数
VISION_DRY_RUN=false              # true 只识别不发送
```

发送机制：使用 AppleScript UI scripting（`tell process "WeChat" ... click at {x,y}`）在微信进程内部点击输入框、粘贴、按回车，可绕过 hardened runtime 对全局模拟事件的拦截。

启动后如果微信窗口还没打开，VisionGateway 会最多等待 `VISION_WINDOW_WAIT_TIMEOUT` 秒（默认 30 秒）并每秒重试；超时仍找不到窗口才会报错。环境变量或 `.env` 中可调整该值。

如果提示找不到窗口，先确认微信聊天窗口已打开且可见，再用下面命令查看实际标题：

```bash
python - <<'PY'
import Quartz
windows = Quartz.CGWindowListCopyWindowInfo(
    Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
    Quartz.kCGNullWindowID,
)
for info in windows:
    title = info.get(Quartz.kCGWindowName, '') or ''
    owner = info.get(Quartz.kCGWindowOwnerName, '') or ''
    if 'WeChat' in owner or '微信' in owner or '微信' in title or 'WeChat' in title:
        print(f"owner={owner!r}, title={title!r}")
PY
```

## 聊天记录格式示例

### JSON

```json
[
  {
    "sender_id": "friend_001",
    "sender_name": "Alice",
    "content": "周末有空吃饭吗？",
    "timestamp": "2026-09-20T10:00:00+00:00"
  },
  {
    "sender_id": "me",
    "sender_name": "我",
    "content": "可以啊，想吃啥"
  }
]
```

### TXT

```text
Alice: 周末有空吃饭吗？
我: 可以啊，想吃啥
Alice: 那一起去吃火锅吧
```

### CSV

```csv
sender_id,sender_name,content,timestamp
friend_001,Alice,周末有空吃饭吗？,2026-09-20T10:00:00+00:00
me,我,可以啊，想吃啥,2026-09-20T10:01:00+00:00
```

## 隐私说明

- 所有画像、会话记忆默认保存在本地 (`./data/profiles`、`./data/chroma`)
- PDF 解析在本地提取文字；图片解析会把图片 base64 传给配置的大模型 vision 接口
- 开启 `VisionGateway` 后，微信聊天窗口截图会传给配置的多模态 LLM 进行消息识别；回复通过本地 AppleScript UI scripting 发送
- LLM 仅接收解析后的文本片段或截图，不会收到完整聊天记录备份
- 请勿将 `.env`、画像文件或聊天记录提交到仓库

## 测试

```bash
pytest
ruff check .
mypy src/
```

## 目录结构

```
src/wechat_agent/
├── access_layer/               # 消息网关
│   ├── base_gateway.py
│   ├── mock_gateway.py
│   └── vision/                 # 视觉监控后端（macOS Quartz + AppleScript UI scripting）
│       ├── backends/
│       ├── extraction.py
│       ├── gateway.py
│       └── models.py
├── agents/                     # 业务 Agent
│   ├── auto_reply.py
│   ├── profile_builder.py
│   ├── safety_guard.py
│   └── session_manager.py
├── ingestion/                  # 聊天记录摄入与解析
│   ├── __init__.py
│   ├── importer.py
│   └── parsers.py
├── llm/                        # 多厂商模型封装
│   ├── providers.py
│   └── router.py
├── memory/                     # 会话/画像存储
│   ├── profile_store.py
│   └── session_store.py
├── models/                     # 数据模型
│   ├── message.py
│   └── state.py
├── orchestration/              # LangGraph 编排
│   └── graph.py
├── tools/                      # 检索工具
│   └── retrieval.py
├── config.py                   # 配置
└── main.py                     # CLI 入口

scripts/
├── import_chat_history.py      # 导入聊天记录并生成画像
├── test_send_wechat.py         # Mock 端到端测试脚本
├── test_vision_friend.py       # 视觉监控 dry-run 测试
├── run_vision_friend.py        # 视觉监控真实发送脚本
├── run_vision.sh               # 视觉监控一键启动 bash 脚本
└── run_e2e.sh                  # 完整端到端：导入 + 生成画像 + 启动监控
```