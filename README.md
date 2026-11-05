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

#### Presets (1–8: tap = apply, hold = save)

A preset holds the whole setup: the look (keying, text/shadow colors, box,
caption layout, output resolution/fps/position) and the configuration
(input type, NDI source or audio device/channel, spoken and caption
languages, recognition engine and model, names/terms, voice settings and
the NDI output name). The API keys (Azure, Soniox, Speechmatics, Claude)
and the Azure region are app-wide and never stored in a preset.

- Applied while **stopped**, everything changes at once.
- Applied while **running**, the look changes immediately; the input,
  recognition, voice and output name are applied when you press Stop (so
  the broadcast isn't cut), and take effect on the next Start.
- Presets saved by older versions hold only the look, and apply only that.

The last session (including the NDI source and output name) is restored
when the app opens. An NDI source that isn't on the network yet is
selected as soon as it appears.

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
  card, Dante Virtual Soundcard, ...), or the audio of any **NDI source**
  on the network (listed as `NDI: …`). NDI audio is received directly
  (audio only), with no virtual audio device needed. *Mix (1+2)* averages
  the first two channels; pick a single channel for multichannel inputs.
  The level meter (on the card and in the panel, in dB) shows the signal
  even before Start.
- **Spoken language:** English, Português, Español, or *Auto* (detected per
  sentence among the three). With Soniox: 60 languages, and *Auto* detects
  any of them.
- **Subtitles in:** Português or English. With Soniox (or Claude
  translating): the 43 languages written left to right in the Latin,
  Cyrillic or Greek alphabet, which the caption font can draw — not Arabic,
  Hebrew, Persian, Urdu, Chinese, Japanese, Korean, Thai or the Indian
  languages (they can still be spoken, or be the voice's language). Speech
  in another language is translated.
- **Engine:**
  - *Local (Whisper, offline)*: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
    for recognition plus offline [Argos Translate](https://github.com/argosopentech/argos-translate)
    models for translation. `auto` uses `large-v3-turbo` on an NVIDIA GPU
    and `base` on the CPU (`small` is more accurate, but on a CPU it runs
    the captions 3.5–6 s behind). The status bar shows the time per
    recognition pass ("s/pass") so you can see whether the machine keeps up. The models download once, on first Start
    (Whisper ~1.6 GB for turbo, ~100 MB per translation pair), into
    `~/.alphasubs_models`. After that it works without internet.
  - *Cloud (Azure Speech)*: Azure AI Speech translation (streaming, lower
    latency, Portuguese from Portugal). It needs a Speech resource key and
    region, and an internet connection. The key is stored in the local
    presets file (`~/.alphasubs_presets.json`). Sentences are split by
    meaning (semantic segmentation), so fast continuous conversation is
    captioned sentence by sentence too. Short fragments left at long
    hesitations are joined to the next sentence. Captions typically arrive
    0.5–1 s after the end of each sentence.
  - *Cloud (Soniox)*: streaming recognition and translation over a
    WebSocket ([soniox.com](https://soniox.com), API key). Measured 0.5–1 s
    from the end of a sentence to its caption, natural Portuguese from
    Portugal. Billed by the minute while running — press Stop when done.
  - *Cloud (Speechmatics)*: streaming recognition (EU endpoint, API key);
    very accurate, ~2–2.5 s per sentence. It can't detect the language in
    real time (choose the spoken language), and its own Portuguese
    translation is Brazilian — use Claude for European Portuguese.
  - With the cloud engines, the app reconnects by itself if the connection
    drops.
- **Translation:** *By the engine*, or *Claude (with context, pt-PT)*: the
  engine only transcribes, and Claude (Anthropic API key, from
  platform.claude.com — a claude.ai subscription doesn't include it)
  translates each sentence with the previous ones as context, into
  European Portuguese. Adds ~0.5–1.5 s per sentence; shows finished
  sentences only. If Claude fails, the engine's own text is used.
- **Names / terms:** comma-separated words to bias recognition (names of
  performers, works, institutions).

The TEXT panel then shows the caption layout (font size, number of lines,
line width, bottom margin, clear-after-silence time) instead of the keying
controls. Shadow, text color, the background box and the OUTPUT settings
work exactly as in the NDI mode, and so do presets.

**Display** (TEXT panel) chooses how captions appear:

- *Finished sentences* (default): each sentence appears on its own, and
  never changes once it's up. A sentence longer than the number of lines is
  shown as consecutive pages, each up for a reading time (faster when more
  text is waiting).
- *Continuous box*: text is written in word by word as the engine confirms
  it, filling the lines from the top; when the box is full it clears and
  goes on. Nothing on screen is ever corrected. Most fluid with Soniox,
  which confirms its translation in chunks every few seconds; with other
  engines (or Claude translating) it writes in each finished sentence.
- *Live roll-up*: text appears as the person talks (fastest, but it visibly
  corrects itself).

Expected latency is about 2–3 s from the end of a sentence to its caption
with the local engine on a CPU, less with a GPU or with Azure.

### Translated voice (VOICE module)

In the audio mode, the translated text can also be spoken, while the
captions keep working. Open the **VOICE** card:

- **Voice:** *Off*, *Local (Piper, offline)*, *Cloud (Azure neural)* or
  *Cloud (Soniox)*. Azure and Soniox voices use the keys from the Input
  panel. Soniox voices speak 60 languages (each voice has its own accent:
  Matilde and Duarte for Portuguese from Portugal) and start playing ~0.8 s
  after a sentence arrives.
- **Speaker:** the voices available for the subtitle language. Local voices
  (~65 MB each) download once, on first use.
- **Output device / Device channels:** a sound card output (headphones,
  interpretation system, Dante...), on channels 1+2 or any single channel
  or pair the device has.
- **NDI:** *Output audio, channels …*: the same NDI source then carries the
  captions (video + alpha) and the voice (48 kHz) on the chosen channels —
  the output gets as many audio channels as the highest one used (2 or 4).
  Or *Own NDI source (audio only)*: a separate source named
  "<output name> Voice 1" / "… Voice 2".
- **Voice language:** *Same as subtitles*, or a different one (e.g. captions
  in Portuguese and voice in English). It is translated from the recognized
  original, not from the captions.
- **Speed / Volume:** adjustable live, per voice.

**Two voices (two languages):** the panel has *Voice 1* and *Voice 2*, each
with its own language, speaker and outputs. On the same sound card or NDI
output, give each its own channels — e.g. *channel 1 (left)* and *channel 2
(right)*, or *channels 1+2* and *3+4*. The VOICE card's speaker then plays
voice 1 on the left and voice 2 on the right. Translation into the second
language: Azure and Claude translate into it as well as they do the
subtitles. Soniox translates into one language per connection, so a voice
in a language other than the subtitles' (and other than the speaker's) gets
**a second Soniox connection** on the same audio, translating into its
language — same quality and latency, and billed like the first. With
Speechmatics it uses the offline translation (English, Spanish or
Portuguese only, lower quality).

A voice never repeats what was said in its own language: the listeners
already heard the original.

**Conversation (Soniox):** for two people speaking two languages on the same
signal (e.g. an interview, PT ↔ EN), set *Spoken language* to one of them,
*Subtitles in* to the other, and tick *Both spoken (conversation)*. One
connection then translates each language into the other: the subtitles
stay in one language, and with Voice 1 in PT and Voice 2 in EN, each voice
speaks only the other person's words. With cloned voices (below), each
person can be heard in their own voice.

**Cloned voices (Soniox):** voices created in the Soniox Console (Project →
Voices → Create voice, from 10 s–2 min of clean speech by one person)
appear in the *Speaker* list as "<name> (cloned)", in every language. Only
clone someone with their written consent, and tell the audience the voice
is synthetic.

**Cloud meter** (main window, under Start): how long each cloud connection
has been up in the current run ("Captions 12 min · Voice FR 12 min"; after
Stop, the last run's), and Soniox's spending today (UTC day, re-read every
2 minutes while running and after Stop). The prepaid **balance** needs a
Soniox *organization* API key (console.soniox.com → Organization → API
keys), in the Input panel's *Org key (balance)* field; the normal key
can't read it.
- **Audio buffer (ms):** default 200. It covers the moments when the
  machine is busy (recognition, synthesis, video). If the voice breaks up,
  raise it; it adds that much delay. Applies on Start.

The voice speaks whole sentences, so it starts about 1.5–2.5 s after each
sentence ends. When a backlog builds up during long speeches, it speeds up
progressively (up to +35%), and it skips sentences only if they would start
more than 20 s late. Fast, continuous conversation is still hard to follow
with a synthetic voice: it works best with presentations and speeches.

Keep the voice out of the microphone's reach (don't send it to the room's
PA), or it will be picked up and translated again.

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
src/speech.py       # speech engines: local Whisper, Azure, Soniox, Speechmatics; Claude translation
src/languages.py    # languages offered per engine
src/translator.py   # offline EN/ES/PT translation (Argos models via CTranslate2)
src/captions.py     # caption layout (roll-up lines) and rendering through the keyer
src/voice.py        # translated voice: Piper/Azure/Soniox TTS to an output device and/or NDI audio
src/main.py         # CLI and main loop
src/gui.py          # Tkinter GUI for operating without a terminal
build_windows.bat   # builds dist\AlphaSubs.exe via PyInstaller (run on Windows)
tools/test_source.py  # synthetic NDI source for loopback testing
```
