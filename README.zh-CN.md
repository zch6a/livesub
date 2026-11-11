<div align="center">

# livesub

**电脑里播放或听到的任何声音，都能离线实时生成字幕，还可以顺便翻译。**

<a href="https://github.com/zch6a/livesub/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/zch6a/livesub/actions/workflows/ci.yml/badge.svg"></a>
<img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white">
<img alt="离线运行" src="https://img.shields.io/badge/runs-offline-2ea44f">
<a href="LICENSE"><img alt="MIT 许可证" src="https://img.shields.io/badge/license-MIT-blue"></a>

[English](README.md) · **简体中文**

</div>

---

视频会议、网课直播、游戏、播客……livesub 在你自己的电脑上用 Whisper 识别语音，显示在一条浮在所有窗口之上的字幕条里，下面还可以附上译文。

- **离线。** 语音识别用 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 在本地运行，CPU 或 CUDA 显卡都可以。翻译发给你自己选的、兼容 OpenAI 接口的服务；指向本地模型（Ollama、LM Studio、llama.cpp），数据就不会离开你的电脑。
- **字幕不闪。** 正在识别的词显示为灰色，只有连续两次识别结果一致才变成白色，之后不会再改。
- **系统级。** 可以采集扬声器正在播放的声音（Windows 用 WASAPI 回环，Linux 用 monitor 源），也可以用麦克风。
- **也能出字幕文件。** `livesub file` 让录音走同一套实时流程，输出 SRT 或 WebVTT：原文、译文或双语。

> **状态：0.1，开发中。** 已用真实 Whisper 对文字转语音生成的英文和中文语音做过端到端验证；其余流程由 74 个基于合成音频的测试覆盖。详见 [POLISH_LOG.md](POLISH_LOG.md)。

## 安装

```bash
pip install "livesub[all] @ git+https://github.com/zch6a/livesub"
```

`[all]` 会装上 faster-whisper 和 soundcard。Whisper 模型在第一次使用时下载一次。

**显卡（可选）。** CTranslate2 需要 CUDA 12 的 cuBLAS 和 cuDNN 9。缺了的话 livesub 会提示，并改用 CPU 运行：

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

## 使用

```bash
livesub live                                  # 给电脑正在播放的声音加字幕，悬浮显示
livesub live --source mic --console           # 用麦克风，在终端里显示
livesub live --language en --translate-to zh --llm-url http://localhost:11434/v1 --llm-model qwen2.5:7b
livesub live --save meeting.srt               # 停止时把字幕存下来
livesub file talk.wav -o talk.srt             # 给录音生成字幕
livesub devices
```

字幕条可以拖到任何位置；双击或按 Esc 退出。

### 选哪个模型

实时识别时，识别器重新读一遍当前这句话的速度，必须比这句话变长的速度快。下表是在笔记本 CPU（int8）上用 faster-whisper 读一段 8 秒窗口的实测耗时：

| 模型 | 大小 | CPU 读 8 秒窗口 | CPU 上能实时吗 |
|---|---|---|---|
| `tiny` | 75 MB | 0.33 秒 | 能，准确率最低 |
| `base` | 145 MB | 0.60 秒 | 能（**CPU 上的默认**） |
| `small` | 480 MB | 1.7 秒 | 会落后约 2 秒（**显卡和文件模式的默认**） |
| `medium`、`large-v3`、`turbo` | 1.5–3 GB | — | 只适合显卡 |

用 `--model` 换模型。知道说的是什么语言时，加上 `--language`：可以跳过语言检测，短句也不容易认错。识别中文时，livesub 会引导 Whisper 输出简体字，否则它经常简繁混用。

## 原理

```
音频 ─▶ 重采样到 16 kHz ─▶ 语音检测 ─▶ 语音缓冲 ─(每秒一次)─▶ Whisper
                                                                 │
     字幕条 ◀── 待定的词 ◀── local agreement ◀──── 词和时间戳 ◀─┘
        ▲                         │ 已确定的词
        └── 译文 ◀── 字幕条目 ◀── 断行（两行，支持中日文）
```

- **语音检测。** 一个带自适应底噪的能量检测器决定*什么时候*调用 Whisper。没人说话时不消耗算力，也不会让 Whisper 去转写静音——它最容易在静音上编出文字。
- **Local agreement**（[Macháček 等，2023](https://arxiv.org/abs/2307.14743)）。识别器反复读不断变长的缓冲，两次连续结果一致的词才确定下来。新识别出的词按*文字*、而不是时间戳，和已确定的部分对齐：Whisper 每次给的时间都会漂移，实测中按时间对齐会悄悄丢词。缓冲超过 12 秒时，会在最后一个已确定的词之后截断，所以无论对方连续说多久，识别耗时都有上限。
- **字幕条目。** 在句子结束、停顿或两行写满时结束一条；写满时尽量在逗号处断开，不让句尾几个字孤零零地单独成条。英文在词与词之间断行，中日文可以在任意字之间断行，但不会让收尾的标点出现在行首。

## 开发

```bash
uv venv && uv pip install numpy pytest
uv run pytest
python scripts/mutate.py      # 检验测试本身：植入 32 个 bug，每个都必须让测试失败
```

## 许可证

[MIT](LICENSE)
