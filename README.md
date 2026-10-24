# AlphaSubs (NDI)

Receives an NDI subtitle video signal (black background, white text), keys out
the background, adds a drop shadow to the text, and re-emits the result as a
new NDI source with an alpha channel (BGRA), ready to be composited on another
machine.

## How it works

- **Keying:** each pixel's luminance is mapped directly to the output alpha
  (`--black-level` = luminance that becomes 100% transparent, `--white-level` =
  luminance that becomes 100% opaque). This preserves the text's edge
  anti-aliasing as partial transparency, with no jagged edges.
- **Shadow:** the text's alpha mask is offset (`--shadow-dx/dy`), blurred
  (`--shadow-blur`), and composited *behind* the text at `--shadow-opacity`
  opacity, using premultiplied-space compositing (avoids color fringing at
  edges) with straight-alpha output, NDI BGRA's standard format.
- **Output:** an `NDIlib.VideoFrameV2` with `FOURCC_VIDEO_TYPE_BGRA` is
  re-emitted via `ndi.send_send_video_v2`, becoming a new NDI source on the
  network.

## Installation

The `ndi-python` package already bundles the NDI runtime (`libndi.dylib`) —
**there's no need to install the NDI SDK separately**, just NDI Tools (to
discover sources on the network, already installed on this machine).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### GUI (recommended for operators)

A window with NDI source selection, sliders for every parameter, and a live
preview of the result — no terminal required to operate it:

```bash
python3 src/gui.py
```

On Windows, after building the executable (see the next section), just
double-click `AlphaSubs.exe` — no Python installation needed on the
operator's machine.

#### Building AlphaSubs.exe (Windows)

On a Windows machine with Python 3.10+ (installed from python.org, to make
sure Tkinter is included):

```bat
build_windows.bat
```

This creates `dist\AlphaSubs.exe`, a single self-contained executable (no
Python or venv needed on the target machine). Test it on a "clean" machine
before distributing it, since packaging native libraries (NDI, OpenCV) can
sometimes need tweaks — see the comments in `build_windows.bat` itself.

### Command line

List the NDI sources available on the network:

```bash
python3 src/main.py --list-sources
```

Run the pipeline (set `--source` to a substring of the input source's name):

```bash
python3 src/main.py --source "CAPTION-PC" --output-name "AlphaSubs"
```

On the destination machine, open NDI Video Monitor (or your compositor/
switcher) and select the `AlphaSubs` source — the alpha channel will be
honored by any NDI receiver that supports BGRA (OBS with the NDI plugin,
vMix, NDI Video Monitor, etc).

### Key parameters

| Flag | Default | Description |
|---|---|---|
| `--black-level` | 8 | Luminance (0-255) treated as pure background |
| `--white-level` | 235 | Luminance (0-255) treated as full text |
| `--shadow-dx` / `--shadow-dy` | 3 / 3 | Shadow offset in pixels |
| `--shadow-blur` | 4 | Shadow Gaussian blur sigma |
| `--shadow-opacity` | 0.6 | Maximum shadow opacity (0-1) |
| `--shadow-color` | `0,0,0` | Shadow color in BGR |

If the received signal's background isn't pure black (e.g. it went through
compression and "black" sits around luminance 16-20), raise `--black-level`
until the background fully disappears without eating into the text's edges.

## Audio mode (speech-to-text captions)

Instead of keying an NDI caption signal, AlphaSubs can listen to an audio
input and generate the captions itself. In the GUI, open the first module
card (**Input**) and set **Input type** to *Audio (speech-to-text)*:

- **Audio device / Channel:** any input device on the computer (mic, sound
  card, Dante Virtual Soundcard, NDI Audio, ...). *Mix (1+2)* averages the
  first two channels; pick a single channel for multichannel interfaces.
  The level meter shows what is being received.
- **Spoken language:** English, Português, Español, or *Auto* (detected per
  sentence among the three).
- **Subtitles in:** Português or English. Speech in another language is
  translated.
- **Engine:**
  - *Local (Whisper, offline)*: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
    for recognition plus offline [Argos Translate](https://github.com/argosopentech/argos-translate)
    models for translation. `auto` uses `large-v3-turbo` on an NVIDIA GPU
    and `small` on the CPU. The models download once, on first Start
    (Whisper ~1.6 GB for turbo, ~100 MB per translation pair), into
    `~/.alphasubs_models`. After that it works without internet.
  - *Cloud (Azure Speech)*: Azure AI Speech translation (streaming, lower
    latency, Portuguese from Portugal). It needs a Speech resource key and
    region, and an internet connection. The key is stored in the local
    presets file (`~/.alphasubs_presets.json`).
- **Names / terms:** comma-separated words to bias recognition (names of
  performers, works, institutions).

The TEXT panel then shows the caption layout (font size, number of roll-up
lines, line width, bottom margin, clear-after-silence time) instead of the
keying controls. Shadow, text color, the background box and the OUTPUT
settings work exactly as in the NDI mode, and so do presets.

Expected latency is about 1–3 s from speech to caption with the local engine
(the text appears while the person is still speaking and is refined when
they pause).

**GPU on Windows:** `requirements.txt` installs the CUDA 12/cuDNN 9 runtime
(`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`), which the app picks up
automatically. If the GPU can't be used, it logs why and falls back to the
CPU. For `AlphaSubs.exe`, see the note in `build_windows.bat`.

## Testing without a real source

`tools/test_source.py` emits a synthetic `TEST-CAPTION-SRC` source (black
background + white block) to validate the pipeline end to end:

```bash
# terminal 1
python3 tools/test_source.py

# terminal 2
python3 src/main.py --source TEST-CAPTION-SRC --output-name "AlphaSubs"
```

## Layout

```
src/ndi_io.py       # thin wrapper over NDIlib: discovery, receiver, sender
src/compositor.py   # luminance keying + shadow (pure numpy/OpenCV, testable in isolation)
src/audio_io.py     # audio input devices + capture (16 kHz mono)
src/speech.py       # speech engines: local Whisper and Azure Speech
src/translator.py   # offline EN/ES/PT translation (Argos models via CTranslate2)
src/captions.py     # caption layout (roll-up lines) and rendering through the keyer
src/main.py         # CLI and main loop
src/gui.py          # Tkinter GUI for operating without a terminal
build_windows.bat   # builds dist\AlphaSubs.exe via PyInstaller (run on Windows)
tools/test_source.py  # synthetic NDI source for loopback testing
```
