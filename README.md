# 语音助手客户端

一个集成实时语音助手，连接到 FunASR STT 服务，实现完整的语音对话流程。

## ✨ 特性

- 🎤 **实时语音识别**：连接到 FunASR STT 服务
- 🤖 **流式 LLM 生成**：支持大模型流式输出
- 🔊 **流式 TTS 合成**：边生成边播放
- 🛑 **支持打断**：播放时可随时打断
- 🎯 **本地 VAD**：自动检测说话开始/结束

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install pyaudio websockets
```

### 2. 启动语音助手

```bash
python voice_assistant.py --stt-host 192.168.8.250 --stt-port 10096
```

### 3. 开始使用

- 对着麦克风说话
- 系统会识别并回复
- 回复时可随时打断

## ⚙️ 配置选项

```bash
python voice_assistant.py \
    --stt-host 192.168.8.250 \    # STT 服务地址
    --stt-port 10096 \              # STT 服务端口
    --stt-mode offline \            # STT 模式（可选：online/offline/2pass）
    --no-ssl \                      # 不使用 SSL（可选）
    --vad-threshold 0.02 \          # VAD 阈值（环境吵可调高）
    --barge-in-threshold 0.025      # 打断阈值（避免误打断可调高）
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--stt-host` | 192.168.8.250 | STT 服务地址 |
| `--stt-port` | 10096 | STT 服务端口 |
| `--stt-mode` | offline | STT 模式（online/offline/2pass） |
| `--no-ssl` | False | 禁用 SSL 连接 |
| `--vad-threshold` | 0.02 | VAD 语音检测阈值（0-1） |
| `--barge-in-threshold` | 0.025 | 打断检测阈值（0-1） |

## 📁 项目结构

```
demo/
├── voice_assistant.py          # 主程序
├── main.py                     # 入口文件
├── requirements.txt            # 依赖列表
└── README.md                   # 本文档
```

## 🔧 接入真实服务

### 接入 LLM

编辑 `voice_assistant.py` 中的 `LLMProvider.generate_stream()` 方法：

```python
async def generate_stream(self, prompt: str, history: list = None):
    # 示例：接入 OpenAI
    import openai
    client = openai.AsyncOpenAI(api_key="your-api-key")

    async for chunk in await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    ):
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

### 接入 TTS

编辑 `voice_assistant.py` 中的 `TTSProvider.synthesize_stream()` 方法：

```python
async def synthesize_stream(self, text: str):
    # 示例：接入 Edge TTS（免费）
    import edge_tts

    communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]
```

## 🐛 常见问题

### STT 模式说明

- **offline**（默认）：离线模式，只在说话结束后返回结果，准确度高
- **online**：在线流式模式，实时返回识别结果，但可能不够准确
- **2pass**：两阶段模式，先返回 online 结果，再返回 offline 修正结果

选择建议：
- 追求准确度 → 使用 `offline`
- 追求实时性 → 使用 `online`
- 两者兼顾 → 使用 `2pass`

### 连接失败

### VAD 误触发（太敏感）

```bash
# 提高阈值
python voice_assistant.py --vad-threshold 0.03
```

### 打断太敏感

```bash
# 提高打断阈值
python voice_assistant.py --barge-in-threshold 0.035
```

### 麦克风无声音

1. 检查麦克风是否连接
2. 检查系统音量设置
3. 确认防火墙没有阻止

## 💡 使用技巧

### 自然对话

- 不要刻意停顿，VAD 会自动检测
- 说完话后自然停顿 0.5-1 秒
- 系统会自动识别结束

### 有效打断

- 在系统回复时直接说话
- 不需要等待，随时可以打断
- 打断后立即进入下一轮识别

## 📖 工作原理

```
麦克风 → VAD检测 → STT服务 → LLM生成 → TTS合成 → 扬声器
   ↑_____________打断_____________|
```

### 状态机

```
IDLE (空闲)
  ↓ 检测到语音
LISTENING (监听)
  ↓ 开始说话
RECOGNIZING (识别中)
  ↓ STT 返回结果
THINKING (LLM生成中)
  ↓ 开始 TTS
SPEAKING (播放中)
  ↓ 检测到打断 / 播放完成
INTERRUPTING → RECOGNIZING
```

## 📝 依赖说明

- **pyaudio**: 音频采集和播放
- **websockets**: 与 STT 服务通信

安装依赖：

```bash
pip install pyaudio websockets
```

## 📄 License

MIT
