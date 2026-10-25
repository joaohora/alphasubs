"""Translated voice (speech-to-speech) for the audio mode.

Each final caption sentence is synthesized (Piper offline, or Azure neural
voices) and played on an audio output device and/or sent as the audio of the
NDI output — alongside the captions, which keep working as before.

A voice can only speak whole sentences (unlike captions, which can show a
partial and correct it), so it runs a few seconds behind the speaker. To
keep that lag from growing during long speeches, it speaks faster when a
backlog builds up and skips sentences that are already too old."""

import queue
import shutil
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape

import numpy as np

NDI_RATE = 48000
PIPER_THREADS = 2
# The voice's output buffer (VoiceTuning.buffer_ms) has to absorb the moments
# when the audio threads can't run: in one Python process, recognition,
# synthesis, the video output and the UI each hold the interpreter (GIL) for
# 20-60 ms at times, occasionally more. Device output: requested as the
# stream latency. NDI: the audio frame size, with SDK clocking — the SDK
# sends each frame exactly one frame-duration after the previous one, so the
# thread has a whole frame of slack to deliver the next, and receivers see
# evenly timestamped audio.
BUFFER_MS_DEFAULT = 200
VOICES_DIR = Path.home() / ".alphasubs_models" / "voices"
PIPER_VOICES_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
ESPEAK_MAX_PATH = 120
PIPER_VOICES = {
    "pt": ("pt_PT-tugão-medium",),
    "en": ("en_GB-alan-medium", "en_GB-cori-medium", "en_US-lessac-medium", "en_US-ryan-medium"),
}
AZURE_VOICES = {
    "pt": ("pt-PT-RaquelNeural", "pt-PT-DuarteNeural", "pt-PT-FernandaNeural"),
    "en": ("en-GB-SoniaNeural", "en-GB-RyanNeural", "en-US-JennyNeural", "en-US-GuyNeural"),
}
# Natural, continuous speech is faster than a synthetic voice, so the voice
# speeds up progressively with the pending backlog: +SPEEDUP_PER_S per second
# of backlog above SPEEDUP_START_S, up to SPEEDUP_MAX. Only a sentence that
# would still start more than MAX_LAG_S late is skipped.
SPEEDUP_START_S = 2.0
SPEEDUP_PER_S = 0.06
SPEEDUP_MAX = 1.35
MAX_LAG_S = 20.0
CHARS_PER_S = 15.0  # rough speaking rate, to estimate not-yet-synthesized text


def voices_for(engine, lang):
    table = AZURE_VOICES if engine == "azure" else PIPER_VOICES
    return table.get(lang, ())


@dataclass
class VoiceSettings:
    engine: str = "off"  # "off" | "local" | "azure"
    voice: str = ""
    # Language the voice speaks (may differ from the captions'). When the
    # recognizer didn't already provide it, the source text is translated
    # offline; `source_langs` are the spoken languages to preload for that.
    lang: str = "pt"
    source_langs: tuple = ()
    device_index: int = None  # audio output device, or None
    ndi: bool = False  # also send the voice as the NDI output's audio
    azure_key: str = ""
    azure_region: str = ""


@dataclass
class VoiceTuning:
    """Bound to GUI controls. speed/volume/muted apply live; buffer_ms on
    Start."""
    speed: float = 1.0
    volume: float = 1.0
    buffer_ms: float = BUFFER_MS_DEFAULT
    # The app's Mute: silences the voice at once (mid-sentence too), and
    # sentences spoken meanwhile are dropped rather than played later.
    muted: bool = False


def _resample(x, in_rate, out_rate):
    """High-quality one-shot resample (libswresample via PyAV, which
    faster-whisper already depends on), with a linear fallback."""
    if in_rate == out_rate or not len(x):
        return x
    try:
        import av

        frame = av.AudioFrame.from_ndarray(x.astype(np.float32)[None, :], format="flt", layout="mono")
        frame.sample_rate = in_rate
        resampler = av.AudioResampler(format="flt", layout="mono", rate=out_rate)
        frames = resampler.resample(frame) + resampler.resample(None)
        return np.concatenate([f.to_ndarray().reshape(-1) for f in frames]).astype(np.float32)
    except Exception:
        n_out = int(round(len(x) * out_rate / in_rate))
        return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x).astype(np.float32)


def _download_piper_voice(name, log):
    """Fetches <name>.onnx + .onnx.json from the Piper voices repository.
    (piper's own downloader can't handle non-ASCII names like 'tugão'.)"""
    lang_code, speaker, quality = name.split("-", 2)
    base = PIPER_VOICES_URL + quote(f"{lang_code.split('_')[0]}/{lang_code}/{speaker}/{quality}/{name}")
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Downloading voice '{name}' (one-time, ~65 MB)...")
    for ext in (".onnx.json", ".onnx"):
        target = VOICES_DIR / f"{name}{ext}"
        partial = target.with_name(target.name + ".part")
        with urllib.request.urlopen(base + ext, timeout=60) as resp, open(partial, "wb") as f:
            shutil.copyfileobj(resp, f)
        partial.replace(target)


def _espeak_data_dir():
    """espeak-ng (Piper's phonemizer) silently falls back to a bogus built-in
    path when its data path is longer than ~160 chars (deep venvs, OneDrive
    folders...). In that case, use a copy under the (short) home folder."""
    from piper.phonemize_espeak import ESPEAK_DATA_DIR

    if len(str(ESPEAK_DATA_DIR)) <= ESPEAK_MAX_PATH:
        return ESPEAK_DATA_DIR
    local = VOICES_DIR.parent / "espeak-ng-data"
    if not (local / "phontab").is_file():
        shutil.copytree(ESPEAK_DATA_DIR, local, dirs_exist_ok=True)
    return local


class _PiperSynth:
    def __init__(self, name, log):
        from piper import PiperVoice

        model = VOICES_DIR / f"{name}.onnx"
        if not model.is_file() or not model.with_suffix(".onnx.json").is_file():
            _download_piper_voice(name, log)
        self._voice = PiperVoice.load(str(model), espeak_data_dir=_espeak_data_dir())
        # Piper's ONNX session uses every core by default; during each
        # sentence's synthesis that starved the audio output threads (gaps
        # heard as cuts). Two threads is still ~20x faster than real time.
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = PIPER_THREADS
        opts.inter_op_num_threads = 1
        self._voice.session = onnxruntime.InferenceSession(
            str(model), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.rate = self._voice.config.sample_rate

    def synth(self, text, speed):
        from piper.config import SynthesisConfig

        cfg = SynthesisConfig(length_scale=1.0 / max(0.5, speed))
        chunks = [c.audio_float_array for c in self._voice.synthesize(text, cfg)]
        return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


class _AzureSynth:
    rate = 24000

    def __init__(self, name, key, region):
        import azure.cognitiveservices.speech as speechsdk

        if not key.strip() or not region.strip():
            raise RuntimeError("Azure voices need the Azure key and region (Input panel).")
        self._sdk = speechsdk
        config = speechsdk.SpeechConfig(subscription=key.strip(), region=region.strip())
        config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm)
        self._synth = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=None)
        self._name = name

    def synth(self, text, speed):
        rate_pct = int(round((speed - 1.0) * 100))
        ssml = (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{self._name[:5]}">'
            f'<voice name="{self._name}"><prosody rate="{rate_pct:+d}%">{escape(text)}</prosody></voice></speak>'
        )
        result = self._synth.speak_ssml_async(ssml).get()
        if result.reason != self._sdk.ResultReason.SynthesizingAudioCompleted:
            details = getattr(result, "cancellation_details", None)
            raise RuntimeError(getattr(details, "error_details", None) or f"Azure TTS failed ({result.reason}).")
        return np.frombuffer(result.audio_data, dtype=np.int16).astype(np.float32) / 32768.0


class _AudioBuffer:
    """FIFO of mono float32 samples for one output, at that output's rate."""

    def __init__(self, rate):
        self.rate = rate
        self._lock = threading.Lock()
        self._chunks = []
        self._len = 0

    def append(self, samples):
        with self._lock:
            self._chunks.append(samples)
            self._len += len(samples)

    def seconds(self):
        return self._len / self.rate

    def read(self, n):
        out = np.zeros(n, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < n and self._chunks:
                chunk = self._chunks[0]
                take = min(n - filled, len(chunk))
                out[filled:filled + take] = chunk[:take]
                filled += take
                if take == len(chunk):
                    self._chunks.pop(0)
                else:
                    self._chunks[0] = chunk[take:]
            self._len -= filled
        return out


class VoiceOutput:
    def __init__(self, settings, tuning, ndi_sender, log):
        self.settings = settings
        self.tuning = tuning
        self.log = log
        self._buffer_s = max(0.02, tuning.buffer_ms / 1000.0)
        self._sender = ndi_sender if settings.ndi else None
        self._texts = queue.Queue()
        self._buffers = []
        self._stream = None
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._synth_loop, daemon=True)]
        if self._sender is not None:
            self._ndi_buf = _AudioBuffer(NDI_RATE)
            self._buffers.append(self._ndi_buf)
            self._threads.append(threading.Thread(target=self._ndi_loop, daemon=True))

    def start(self):
        s = self.settings
        if s.device_index is not None:
            import sounddevice as sd

            info = sd.query_devices(s.device_index)
            rate = int(info["default_samplerate"])
            self._dev_buf = _AudioBuffer(rate)
            self._buffers.append(self._dev_buf)
            channels = min(2, int(info["max_output_channels"]))
            self._stream = sd.OutputStream(
                device=s.device_index, samplerate=rate, channels=channels, dtype="float32",
                latency=self._buffer_s, callback=self._device_callback,
            )
            self._stream.start()
        for t in self._threads:
            t.start()

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3.0)
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def say(self, info):
        """Queues one final sentence: the `info` dict from the speech
        engine (see speech.py), so the voice can speak its own language."""
        self._texts.put((time.monotonic(), info))

    def _pending_s(self):
        audio = max((b.seconds() for b in self._buffers), default=0.0)
        with self._texts.mutex:
            queued = sum(len(info.get("source", "")) for _t, info in self._texts.queue)
        return audio + queued / CHARS_PER_S

    def _translator(self, src):
        if src not in self._translators:
            from translator import Translator

            self._translators[src] = Translator(src, self.settings.lang, log=self.log)
        return self._translators[src]

    def _text_for(self, info):
        """The sentence in the voice's language: as provided by the
        recognizer, the source itself, or translated offline from it."""
        lang = self.settings.lang
        text = info.get("texts", {}).get(lang)
        if text:
            return text
        src, source = info.get("lang"), info.get("source", "")
        if not source:
            return ""
        if src == lang:
            return source
        return self._translator(src).translate(source)

    def _synth_loop(self):
        s = self.settings
        self._translators = {}
        try:
            if s.engine == "azure":
                synth = _AzureSynth(s.voice, s.azure_key, s.azure_region)
            else:
                synth = _PiperSynth(s.voice, self.log)
            for src in s.source_langs:
                if src != s.lang:
                    self._translator(src)  # load (and download) now, not on the first sentence
            self.log(f"Voice ready ({s.voice}, speaking {s.lang}).")
        except Exception as e:
            self.log(f"Voice error: {e}")
            return
        while not self._stop.is_set():
            try:
                queued_at, info = self._texts.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.tuning.muted:
                continue
            waited = time.monotonic() - queued_at
            buffered = max((b.seconds() for b in self._buffers), default=0.0)
            if waited + buffered > MAX_LAG_S:
                self.log(f"Voice too far behind — skipped: {info.get('source', '')[:60]}")
                continue
            backlog = max(0.0, self._pending_s() - SPEEDUP_START_S)
            speed = self.tuning.speed * min(SPEEDUP_MAX, 1.0 + backlog * SPEEDUP_PER_S)
            try:
                text = self._text_for(info)
                if not text:
                    continue
                pcm = synth.synth(text, speed)
            except Exception as e:
                self.log(f"Voice error: {e}")
                continue
            for buf in self._buffers:
                buf.append(_resample(pcm, synth.rate, buf.rate))

    def _gain(self):
        return 0.0 if self.tuning.muted else self.tuning.volume

    def _device_callback(self, outdata, frames, _time_info, _status):
        block = self._dev_buf.read(frames) * self._gain()
        outdata[:] = np.clip(block, -1.0, 1.0)[:, None]

    def _ndi_loop(self):
        """Sends a continuous 48 kHz stereo stream (silence between
        sentences) in frames of the buffer size. With SDK clocking each
        send blocks until its slot, which paces the loop; without it (old
        NDI bindings), the loop paces itself."""
        n = int(NDI_RATE * self._buffer_s)
        next_t = time.perf_counter()
        while not self._stop.is_set():
            block = np.clip(self._ndi_buf.read(n) * self._gain(), -1.0, 1.0)
            try:
                self._sender.send_audio(np.vstack([block, block]), NDI_RATE)
            except Exception as e:
                self.log(f"NDI audio error: {e}")
                return
            if not self._sender.audio_clocked:
                next_t += n / NDI_RATE
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.5:
                    next_t = time.perf_counter()
