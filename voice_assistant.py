# -*- encoding: utf-8 -*-
"""
全双工实时语音助手 - 集成客户端
直接与 STT/LLM/TTS 服务通信，支持打断

使用流程：
1. 连接到 FunASR STT 服务 (192.168.8.250:10096)
2. 采集麦克风音频并发送给 STT
3. 接收识别结果 → 调用 LLM → 调用 TTS → 播放
4. 支持播放时随时打断
"""

from __future__ import annotations

import asyncio
import json
import time
import struct
import base64
from typing import Optional, AsyncIterator
from dataclasses import dataclass
from enum import Enum, auto
from collections import deque

import websockets
import ssl
import pyaudio


class SessionState(Enum):
    """会话状态"""
    IDLE = auto()           # 空闲
    LISTENING = auto()      # 监听中
    RECOGNIZING = auto()    # 识别中
    THINKING = auto()       # LLM 生成中
    SPEAKING = auto()       # TTS 播放中
    INTERRUPTING = auto()   # 打断处理中


@dataclass
class AudioConfig:
    """音频配置"""
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # 16bit
    chunk_size: int = 3200  # PyAudio 缓冲区大小 (200ms)
    frame_duration_ms: int = 20  # STT 帧长


@dataclass
class VADConfig:
    """VAD 配置"""
    speech_start_threshold: float = 0.02      # 语音开始阈值
    speech_end_threshold_ms: int = 600        # 语音结束静音时长 (ms)
    barge_in_threshold: float = 0.025         # 打断阈值


class SimpleVAD:
    """简单 VAD 实现"""

    def __init__(self, config: VADConfig):
        self.config = config
        self.is_speech = False
        self.speech_start_time: Optional[float] = None
        self.speech_end_time: Optional[float] = None
        self.last_speech_time: Optional[float] = None

    def process(self, audio_frame: bytes) -> tuple[bool, Optional[str]]:
        """
        处理音频帧
        返回: (是否是语音, 事件: "speech_start" | "speech_end" | None)
        """
        energy = self._calculate_energy(audio_frame)
        now = time.time()

        event = None

        if energy > self.config.speech_start_threshold:
            self.last_speech_time = now

            if not self.is_speech:
                if self.speech_start_time is None:
                    self.speech_start_time = now
                elif now - self.speech_start_time >= 0.2:  # 200ms 持续语音
                    self.is_speech = True
                    event = "speech_start"
            else:
                self.speech_end_time = None
        else:
            if self.is_speech:
                if self.speech_end_time is None:
                    self.speech_end_time = now
                elif (now - self.speech_end_time >=
                      self.config.speech_end_threshold_ms / 1000):
                    self.is_speech = False
                    self.speech_start_time = None
                    self.speech_end_time = None
                    event = "speech_end"

        return self.is_speech, event

    def check_barge_in(self, audio_frame: bytes) -> bool:
        """检查是否应该打断"""
        energy = self._calculate_energy(audio_frame)
        return energy > self.config.barge_in_threshold

    def _calculate_energy(self, frame: bytes) -> float:
        """计算音频帧能量 (RMS)"""
        samples = struct.unpack(f"<{len(frame)//2}h", frame)
        if not samples:
            return 0.0
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        return rms / 32768.0

    def reset(self):
        """重置状态"""
        self.is_speech = False
        self.speech_start_time = None
        self.speech_end_time = None
        self.last_speech_time = None


class AudioPlayer:
    """音频播放器 - 支持流式播放和立即停止"""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.is_playing = False
        self.current_turn_id = 0
        self._stop_requested = False

    def start(self):
        """启动播放器"""
        if self.stream is None:
            self.stream = self.audio.open(
                format=self.audio.get_format_from_width(self.config.sample_width),
                channels=self.config.channels,
                rate=self.config.sample_rate,
                output=True,
                frames_per_buffer=self.config.chunk_size
            )
            print("[Player] 音频播放器已启动")

    def play(self, audio_data: bytes):
        """播放音频数据"""
        if self.stream and not self._stop_requested:
            self.stream.write(audio_data)
            self.is_playing = True

    async def stop(self):
        """立即停止播放"""
        self._stop_requested = True
        self.is_playing = False
        # 等待一下让当前播放完成
        await asyncio.sleep(0.1)
        self._stop_requested = False
        print("[Player] 播放已停止")

    def close(self):
        """关闭播放器"""
        self._stop_requested = True
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.audio.terminate()


class LLMProvider:
    """LLM 服务提供者 - vLLM API 客户端"""

    def __init__(
        self,
        api_key: str = "dummy-api-key",
        base_url: str = "http://192.168.8.88:8888/v1",
        model: str = "Qwen3.5"
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate_stream(self, prompt: str, history: list = None) -> AsyncIterator[str]:
        """
        调用 vLLM API 生成文本（非流式）
        返回: 完整的回复文本（以单条 yield 返回）
        """
        import aiohttp

        # 构建消息历史
        messages = []
        if history:
            for item in history:
                messages.append(item)
        messages.append({"role": "user", "content": prompt})

        # API 请求体（非流式）
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 1024
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"LLM API 错误 (HTTP {response.status}): {error_text}")

                    data = await response.json()

                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0].get("message", {}).get("content", "")
                        if content:
                            # 非流式输出，直接返回完整内容
                            yield content
                    else:
                        raise Exception(f"LLM API 返回格式异常: {data}")

        except asyncio.TimeoutError:
            raise Exception("LLM API 请求超时")
        except aiohttp.ClientError as e:
            raise Exception(f"LLM API 连接错误: {e}")

    def should_flush_tts(self, buffer: str) -> tuple[bool, str, str]:
        """
        判断是否应该将缓冲区内容送 TTS
        返回: (是否刷新, 刷新内容, 剩余内容)
        """
        import re
        delimiters = r'([，。！？；,.!?;])'
        parts = re.split(delimiters, buffer)

        if len(parts) >= 3:  # 有完整句子
            flush_text = "".join(parts[:3])
            remaining = "".join(parts[3:])
            return True, flush_text, remaining

        # 缓冲区太长也强制刷新
        if len(buffer) >= 20:
            return True, buffer, ""

        return False, "", buffer


class TTSProvider:
    """TTS 服务提供者 - 流式合成"""

    def __init__(self, api_url: str = None):
        self.api_url = api_url

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """
        流式合成音频
        TODO: 替换为真实的 TTS API 调用
        """
        # 模拟 TTS 输出 - 替换为真实 API
        # 每个字生成 10 帧 20ms 音频 = 200ms
        frame_count = len(text) * 10
        for i in range(frame_count):
            await asyncio.sleep(0.02)
            # 生成 20ms 静音 PCM 数据 (16kHz, 16bit, mono)
            yield bytes(640)


class VoiceAssistant:
    """
    语音助手客户端
    集成 STT → LLM → TTS 全流程
    """

    def __init__(
        self,
        stt_host: str = "192.168.8.250",
        stt_port: int = 10096,
        stt_mode: str = "offline",
        use_ssl: bool = True,
        audio_config: AudioConfig = None,
        vad_config: VADConfig = None
    ):
        # STT 服务配置
        self.stt_host = stt_host
        self.stt_port = stt_port
        self.stt_mode = stt_mode
        self.use_ssl = use_ssl

        # 配置
        self.audio_config = audio_config or AudioConfig()
        self.vad_config = vad_config or VADConfig()

        # 状态
        self.state = SessionState.IDLE
        self.turn_id = 0
        self.is_interrupted = False

        # 组件
        self.vad = SimpleVAD(self.vad_config)
        self.player: Optional[AudioPlayer] = None
        self.llm = LLMProvider()
        self.tts = TTSProvider()

        # WebSocket
        self.stt_websocket: Optional[websockets.client.WebSocketClientProtocol] = None
        self.stt_connected = False

        # PyAudio
        self.audio = pyaudio.PyAudio()
        self.mic_stream: Optional[pyaudio.Stream] = None

        # 运行标志
        self.running = False

        # 当前识别文本
        self.current_text = ""
        self.final_text = ""

    async def connect_stt(self):
        """连接到 FunASR STT 服务"""
        uri = f"{'wss' if self.use_ssl else 'ws'}://{self.stt_host}:{self.stt_port}"
        print(f"[STT] 连接到: {uri}")

        if self.use_ssl:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        else:
            ssl_context = None

        self.stt_websocket = await websockets.connect(
            uri,
            subprotocols=["binary"],
            ping_interval=None,
            ssl=ssl_context
        )

        # 发送配置消息
        config = {
            "mode": self.stt_mode,
            "chunk_size": [5, 10, 5],
            "chunk_interval": 10,
            "encoder_chunk_look_back": 4,
            "decoder_chunk_look_back": 0,
            "wav_name": "assistant",
            "is_speaking": True,
            "itn": True,
        }
        await self.stt_websocket.send(json.dumps(config))
        self.stt_connected = True
        print(f"[STT] 配置已发送 (mode={self.stt_mode})")

    async def start(self):
        """启动语音助手"""
        print("=" * 60)
        print("🎤 语音助手启动中...")
        print("=" * 60)

        # 连接 STT
        await self.connect_stt()

        # 启动播放器
        self.player = AudioPlayer(self.audio_config)
        self.player.start()

        # 打开麦克风
        self.mic_stream = self.audio.open(
            format=self.audio.get_format_from_width(self.audio_config.sample_width),
            channels=self.audio_config.channels,
            rate=self.audio_config.sample_rate,
            input=True,
            frames_per_buffer=self.audio_config.chunk_size
        )
        print("[Mic] 麦克风已打开")

        self.running = True
        self.state = SessionState.LISTENING

        print("=" * 60)
        print("✅ 语音助手已就绪")
        print("   - 直接说话即可开始对话")
        print("   - 系统回复时可随时打断")
        print("   - 按 Ctrl+C 退出")
        print("=" * 60)
        print()

    async def run(self):
        """运行主循环"""
        try:
            # 启动多个任务
            tasks = [
                asyncio.create_task(self._audio_capture_loop()),
                asyncio.create_task(self._stt_receive_loop()),
            ]

            await asyncio.gather(*tasks, return_exceptions=True)

        except asyncio.CancelledError:
            pass
        finally:
            await self.close()

    async def _audio_capture_loop(self):
        """音频采集循环"""
        try:
            while self.running:
                # 读取音频
                audio_chunk = self.mic_stream.read(
                    self.audio_config.chunk_size,
                    exception_on_overflow=False
                )

                # VAD 处理
                is_speech, event = self.vad.process(audio_chunk)

                # 检查打断
                if self.player and self.player.is_playing:
                    if self.vad.check_barge_in(audio_chunk):
                        print("\n[Assistant] 🛑 检测到打断！")
                        await self._interrupt()
                        continue

                # 发送音频到 STT
                if self.stt_websocket and self.stt_connected:
                    try:
                        await self.stt_websocket.send(audio_chunk)
                    except Exception as e:
                        print(f"[STT] 发送音频失败: {e}")
                        self.stt_connected = False

                # 处理 VAD 事件
                if event == "speech_start":
                    if self.state == SessionState.IDLE:
                        self.state = SessionState.RECOGNIZING
                        self.current_text = ""
                        print(f"\n[Assistant] 🎤 检测到语音...")

                elif event == "speech_end":
                    print(f" [语音结束]")
                    # 发送说话结束标记
                    if self.stt_websocket and self.stt_connected:
                        try:
                            await self.stt_websocket.send(
                                json.dumps({"is_speaking": False})
                            )
                        except Exception as e:
                            print(f"[STT] 发送结束标记失败: {e}")
                            self.stt_connected = False

        except Exception as e:
            print(f"[AudioCapture] 错误: {e}")
            import traceback
            traceback.print_exc()

    async def _stt_receive_loop(self):
        """STT 结果接收循环"""
        try:
            async for message in self.stt_websocket:
                if self.is_interrupted:
                    continue

                data = json.loads(message)
                self._process_stt_result(data)

        except websockets.exceptions.ConnectionClosed:
            print("[STT] 连接已关闭")
        except Exception as e:
            print(f"[STTReceive] 错误: {e}")
            import traceback
            traceback.print_exc()

    def _process_stt_result(self, data: dict):
        """处理 STT 识别结果"""
        text = data.get("text", "")
        mode = data.get("mode", "")
        is_final = data.get("is_final", False)

        if not text:
            return

        # 更新当前文本
        self.current_text = text

        # 显示识别结果
        if mode in ("online", "2pass-online"):
            # Partial 结果
            print(f"\r[STT] {text}", end="", flush=True)
        else:
            # Final 结果
            print(f"\r[STT] ✅ {text}")
            self.final_text = text

            # 触发 LLM 处理
            if text.strip() and not self.is_interrupted:
                asyncio.create_task(self._process_llm_pipeline(text))

    async def _process_llm_pipeline(self, user_text: str):
        """处理 LLM → TTS 流水线"""
        if self.is_interrupted:
            return

        self.state = SessionState.THINKING
        turn_id = self.turn_id

        print(f"\n[LLM] 🤔 用户: {user_text}")
        print("[LLM] 💭 助手: ", end="", flush=True)

        try:
            text_buffer = ""
            full_response = ""

            # 流式生成
            async for token in self.llm.generate_stream(user_text):
                # 检查是否被打断
                if self.is_interrupted or turn_id != self.turn_id:
                    print(f"\n[LLM] ❌ 生成被中断")
                    return

                text_buffer += token
                full_response += token

                print(token, end="", flush=True)

                # 检查是否该送 TTS
                should_flush, flush_text, remaining = self.llm.should_flush_tts(text_buffer)

                if should_flush and flush_text:
                    # 启动 TTS
                    if not self.is_interrupted:
                        await self._synthesize_and_play(turn_id, flush_text)
                    text_buffer = remaining

            # 处理剩余文本
            if text_buffer and not self.is_interrupted:
                await self._synthesize_and_play(turn_id, text_buffer, is_end=True)

            print()  # 换行

        except asyncio.CancelledError:
            print(f"\n[LLM] ❌ 任务被取消")
        except Exception as e:
            print(f"\n[LLM] ❌ 错误: {e}")
            import traceback
            traceback.print_exc()

    async def _synthesize_and_play(self, turn_id: int, text: str, is_end: bool = False):
        """合成并播放音频"""
        if self.is_interrupted or turn_id != self.turn_id:
            return

        self.state = SessionState.SPEAKING

        try:
            async for audio_chunk in self.tts.synthesize_stream(text):
                # 检查是否被打断
                if self.is_interrupted or turn_id != self.turn_id:
                    print(f"\n[TTS] ❌ 合成被中断")
                    return

                if self.player:
                    self.player.play(audio_chunk)

            if is_end and not self.is_interrupted:
                self.state = SessionState.LISTENING
                await asyncio.sleep(0.5)  # 短暂停顿
                print(f"\n[Assistant] ✅ 回复完成")

        except asyncio.CancelledError:
            print(f"\n[TTS] ❌ 任务被取消")
        except Exception as e:
            print(f"\n[TTS] ❌ 错误: {e}")

    async def _interrupt(self):
        """处理打断"""
        self.is_interrupted = True
        self.turn_id += 1
        self.state = SessionState.INTERRUPTING

        print(f"[Assistant] 🔄 处理打断 (turn {self.turn_id})")

        # 停止播放
        if self.player:
            await self.player.stop()

        # 重置 VAD
        self.vad.reset()

        # 短暂延迟后恢复
        await asyncio.sleep(0.3)

        self.is_interrupted = False
        self.state = SessionState.LISTENING
        self.current_text = ""

        print("[Assistant] ✅ 准备好接收新指令")

    async def close(self):
        """关闭助手"""
        print("\n[Assistant] 关闭中...")
        self.running = False
        self.is_interrupted = True

        # 关闭麦克风
        if self.mic_stream:
            self.mic_stream.stop_stream()
            self.mic_stream.close()
            self.mic_stream = None

        # 关闭播放器
        if self.player:
            self.player.close()
            self.player = None

        # 关闭 STT 连接
        if self.stt_websocket:
            self.stt_connected = False
            try:
                await self.stt_websocket.close()
            except:
                pass
            self.stt_websocket = None

        # 终止 PyAudio
        self.audio.terminate()

        print("[Assistant] 已关闭")


# ==================== 启动入口 ====================

async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="全双工实时语音助手")
    parser.add_argument(
        "--stt-host",
        type=str,
        default="192.168.8.250",
        help="STT 服务地址"
    )
    parser.add_argument(
        "--stt-port",
        type=int,
        default=10096,
        help="STT 服务端口"
    )
    parser.add_argument(
        "--no-ssl",
        action="store_true",
        help="不使用 SSL 连接"
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.02,
        help="VAD 能量阈值"
    )
    parser.add_argument(
        "--barge-in-threshold",
        type=float,
        default=0.025,
        help="打断检测阈值"
    )
    parser.add_argument(
        "--stt-mode",
        type=str,
        default="offline",
        choices=["online", "offline", "2pass"],
        help="STT 模式: online(流式), offline(离线), 2pass(两阶段)"
    )

    args = parser.parse_args()

    # 创建配置
    audio_config = AudioConfig()
    vad_config = VADConfig(
        speech_start_threshold=args.vad_threshold,
        barge_in_threshold=args.barge_in_threshold
    )

    # 创建助手
    assistant = VoiceAssistant(
        stt_host=args.stt_host,
        stt_port=args.stt_port,
        stt_mode=args.stt_mode,
        use_ssl=not args.no_ssl,
        audio_config=audio_config,
        vad_config=vad_config
    )

    try:
        await assistant.start()
        await assistant.run()
    except KeyboardInterrupt:
        print("\n[Assistant] 用户退出")
    finally:
        await assistant.close()


if __name__ == "__main__":
    # 设置 Windows 事件循环策略（如果需要）
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(main())
