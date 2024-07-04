#!/usr/bin/env python
"""
在本地进行录音 + 转写的单脚本代码。不依赖于云服务（e.g., redis, socket），适合于离线使用。

依赖安装:
    pip3 install pyaudio webrtcvad faster-whisper

运行方式:
    python3 local_deploy.py
"""

import collections
import io
import logging
import queue
import threading
import typing
import wave
from io import BytesIO

import codefast as cf
import pyaudio
import webrtcvad
from faster_whisper import WhisperModel

import asyncio
from ollama import AsyncClient
import json

# 解决bug问题
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

history = [{
    'role': 'system',
    'content': 'your name is Conor,and you are a helpful assistant,you are good at coding by python,you always give the right answers step by step!',
}]

logging.basicConfig(level=logging.INFO,
                    format='%(name)s - %(levelname)s - %(message)s')


class Queues:
    audio = queue.Queue()
    text = queue.Queue()


class Transcriber(threading.Thread):
    def __init__(
            self,
            model_size: str,
            device: str = "auto",
            compute_type: str = "default",
            prompt: str = '实时/低延迟语音转写服务，林黛玉、倒拔、杨柳树、鲁迅、周树人、关键词、转写正确') -> None:
        """ FasterWhisper 语音转写

        Args:
            model_size (str): 模型大小，可选项为 "tiny", "base", "small", "medium", "large" 。
                更多信息参考：https://github.com/openai/whisper
            device (str, optional): 模型运行设备。
            compute_type (str, optional): 计算类型。默认为"default"。
            prompt (str, optional): 初始提示。如果需要转写简体中文，可以使用简体中文提示。
        """
        super().__init__()
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.prompt = prompt

    def __enter__(self) -> 'Transcriber':
        try:
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        except Exception as e:
            logging.error("Failed to initialize WhisperModel: %s", e)
            raise e
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    def __call__(self, audio: bytes) -> typing.Generator[str, None, None]:
        if not audio:
            logging.error("Received empty audio data.")
            return
        try:
            segments, info = self._model.transcribe(BytesIO(audio), initial_prompt=self.prompt, vad_filter=True)
            for segment in segments:
                t = segment.text
                if self.prompt in t.strip():
                    continue
                if t.strip().replace('.', ''):
                    yield t
        except Exception as e:
            logging.error("Error during transcription: %s", e)

    def run(self):
        while True:
            audio = Queues.audio.get()
            if audio:
                text = ''
                for seg in self(audio):
                    logging.info(cf.fp.cyan(seg))
                    text += seg
                Queues.text.put(text)


class AudioRecorder(threading.Thread):
    """ Audio recorder.
    Args:
        channels (int, 可选): 通道数，默认为1（单声道）。
        rate (int, 可选): 采样率，默认为16000 Hz。
        chunk (int, 可选): 缓冲区中的帧数，默认为256。
        frame_duration (int, 可选): 每帧的持续时间（单位：毫秒），默认为30。
    """
    def __init__(self,
                 channels: int = 1,
                 sample_rate: int = 16000,
                 chunk: int = 256,
                 frame_duration: int = 30) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.frame_size = (sample_rate * frame_duration // 1000)
        self.__frames: typing.List[bytes] = []

    def __enter__(self) -> 'AudioRecorder':
        try:
            self.vad = webrtcvad.Vad()
            # 设置 VAD 的敏感度。参数是一个 0 到 3 之间的整数。0 表示对非语音最不敏感，3 最敏感。
            self.vad.set_mode(1)

            self.audio = pyaudio.PyAudio()
            self.sample_width = self.audio.get_sample_size(pyaudio.paInt16)
            self.stream = self.audio.open(format=pyaudio.paInt16,
                                          channels=self.channels,
                                          rate=self.sample_rate,
                                          input=True,
                                          frames_per_buffer=self.chunk)
        except Exception as e:
            logging.error("Failed to initialize audio recorder: %s", e)
            raise e
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
        except Exception as e:
            logging.error("Error during cleanup of audio recorder: %s", e)

    def __bytes__(self) -> bytes:
        buf = io.BytesIO()
        try:
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.sample_width)
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.__frames))
                self.__frames.clear()
        except Exception as e:
            logging.error("Error during byte conversion: %s", e)
        return buf.getvalue()

    def run(self):
        """ Record audio until silence is detected.
        """
        MAXLEN = 30
        watcher = collections.deque(maxlen=MAXLEN)
        triggered, ratio = False, 0.5
        while True:
            try:
                frame = self.stream.read(self.frame_size)
                is_speech = self.vad.is_speech(frame, self.sample_rate)
                watcher.append(is_speech)
                self.__frames.append(frame)
                if not triggered:
                    num_voiced = len([x for x in watcher if x])
                    if num_voiced > ratio * watcher.maxlen:
                        logging.info("start recording...")
                        triggered = True
                        watcher.clear()
                        self.__frames = self.__frames[-MAXLEN:]
                else:
                    num_unvoiced = len([x for x in watcher if not x])
                    if num_unvoiced > ratio * watcher.maxlen:
                        logging.info("stop recording...")
                        triggered = False
                        audio_data = bytes(self)
                        if audio_data:
                            Queues.audio.put(audio_data)
                            logging.info("audio task number: {}".format(Queues.audio.qsize()))
            except Exception as e:
                logging.error("Error during audio recording: %s", e)


class Chat(threading.Thread):
    def __init__(self, prompt: str) -> None:
        super().__init__()
        self.prompt = prompt

    async def chat_to_ollama(self, text):
        history.append({'role': 'user', 'content': text})
        answer = ''
        try:
            client = AsyncClient(host='http://124.223.159.146:11434')# replace the host IP with your own ollama serve IP
            message = {'role': 'user', 'content': text}
            async for part in await client.chat(model='deepseek-coder', messages=history, stream=True):#replace the model your own
                print(part['message']['content'], end='', flush=True)
                answer += part['message']['content']
            history.append({'role': 'assistant', 'content': answer})
            with open('history.json', 'w') as f:
                json.dump(history, f)

        except Exception as e:
            logging.error("Error during chat communication: %s", e)

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while True:
            text = Queues.text.get()
            if text:
                loop.run_until_complete(self.chat_to_ollama(text))


def main():
    try:
        with AudioRecorder(channels=1, sample_rate=16000) as recorder:
            with Transcriber(model_size="small") as transcriber:#model_size：tiny，base，small，medium，large，large-v1，large-v2，large-v3
                recorder.start()
                transcriber.start()
                chat = Chat("")
                chat.start()

                recorder.join()
                transcriber.join()
                chat.join()

    except KeyboardInterrupt:
        print("KeyboardInterrupt: terminating...")
    except Exception as e:
        logging.error(e, exc_info=True, stack_info=True)


if __name__ == "__main__":
    main()
