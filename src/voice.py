"""Translated voice (speech-to-speech) for the audio mode.

Each final caption sentence is synthesized (Piper offline, Azure or Soniox)
and played on an audio output device and/or sent as the audio of the NDI
output — alongside the captions, which keep working as before. There can be
two voices, each in its own language, on separate channels of the same
outputs (or on outputs of their own). The original sound (the input) can
play under the voices, lowered while they speak ("ducking"), and the
operator can monitor each voice on this computer's default output.

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
# Soniox voices speak every language, each with its own accent: per
# language, the voices of that origin (judged by their names); other
# languages get a neutral selection.
SONIOX_VOICES = {
    "pt": ("Matilde", "Duarte", "Rui", "Nuno", "Diogo", "Bernardo"),
    "en": ("Victoria", "Daniel", "Oliver", "Isla", "Grace", "Adrian"),
    "es": ("Lucia", "Rafael", "Carmen", "Javier", "Marta", "Sergio", "Alvaro", "Hector"),
    "fr": ("Camille", "Julien", "Elodie", "Olivier", "Manon", "Antoine", "Celine", "Mathieu"),
    "de": ("Lena", "Stefan", "Annika", "Tobias", "Marlene", "Konrad", "Jonas", "Matthias"),
    "it": ("Giulia", "Marco", "Valeria", "Riccardo", "Alessandro", "Davide"),
    "uk": ("Yana", "Taras", "Bohdan"),
    "ru": ("Yana", "Andrei", "Kirill", "Vadim"),
    "pl": ("Marek", "Szymon"),
    "ar": ("Layla", "Faisal", "Rania", "Yousef", "Hind", "Karim"),
    "zh": ("Meilin", "Wei", "Haoran", "Zihan"),
    "ja": ("Aoi", "Kenji", "Yuki", "Haruto", "Sakura", "Sota"),
    "ko": ("Mina", "Jihoon", "Jiwoo", "Minjun"),
    "hi": ("Priya", "Arjun", "Kavya", "Rohan", "Ishita", "Kabir"),
    "vi": ("Linh", "Mai", "Huong"),
    "id": ("Nurul", "Bayu", "Budi", "Rizky"),
    "tl": ("Maricel",),
}
SONIOX_VOICES_OTHER = ("Victoria", "Daniel", "Grace", "Adrian", "Lucia", "Rafael")
SONIOX_TTS_URL = "https://tts-rt.soniox.com/tts"
SONIOX_TTS_MODEL = "tts-rt-v2"
SONIOX_SPEED_RANGE = (0.7, 1.3)
SONIOX_CHUNK_BYTES = 9600  # 0.2 s at 24 kHz, 16-bit
# Natural, continuous speech is faster than a synthetic voice, so the voice
# speeds up progressively with the pending backlog: +SPEEDUP_PER_S per second
# of backlog above SPEEDUP_START_S, up to SPEEDUP_MAX. Only a sentence that
# would still start more than MAX_LAG_S late is skipped.
SPEEDUP_START_S = 2.0
SPEEDUP_PER_S = 0.06
SPEEDUP_MAX = 1.35
MAX_LAG_S = 20.0
CHARS_PER_S = 15.0  # rough speaking rate, to estimate not-yet-synthesized text
# Ducking: the original drops to the chosen level within ~DUCK_ATTACK_S of a
# voice starting and comes back over ~DUCK_RELEASE_S once it has been silent
# for DUCK_HOLD_S (the pauses between sentences and synthesis chunks don't
# make it pump). The original is live audio on another clock: it's played
# with ORIGINAL_PREFILL_S of slack and trimmed past ORIGINAL_MAX_S of backlog.
DUCK_ATTACK_S = 0.04
DUCK_RELEASE_S = 0.35
DUCK_HOLD_S = 0.6
ORIGINAL_PREFILL_S = 0.08
ORIGINAL_MAX_S = 0.35


def _soniox_api(key, method, path, body=None, content_type=None):
    """A call to the Soniox REST API (project key); the JSON reply, or None."""
    import json
    import urllib.error

    headers = {"Authorization": f"Bearer {key.strip()}", "User-Agent": "AlphaSubs"}
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request("https://api.soniox.com" + path, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("message", "")
        except Exception:
            detail = ""
        raise RuntimeError(f"Soniox error {e.code}: {detail or e.reason}")
    return json.loads(data) if data else None


def _tts_status(voice):
    """A cloned voice's state for the TTS model: processing / ready / failed
    (with the reason)."""
    for m in voice.get("models", []):
        if m.get("model") == SONIOX_TTS_MODEL:
            if m.get("status") == "failed":
                return f"failed ({m.get('error_message') or m.get('error_type') or 'unknown'})"
            return m.get("status", "processing")
    return "processing"


def soniox_voices(key):
    """[(name, voice ID, status)] of the voices cloned in the key's Soniox
    project; status as in _tts_status."""
    voices = _soniox_api(key, "GET", "/v1/voices")["voices"]
    return [(v["name"], v["id"], _tts_status(v)) for v in voices]


def soniox_cloned_voices(key):
    """{name: voice ID} of the cloned voices ready to speak."""
    return {name: vid for name, vid, status in soniox_voices(key) if status == "ready"}


def soniox_clone_voice(key, name, wav_path, timeout_s=120):
    """Creates a cloned voice from a reference clip (one speaker, little
    noise, up to 2 minutes) and waits until it can speak. Returns its ID."""
    import uuid

    boundary = uuid.uuid4().hex
    data = Path(wav_path).read_bytes()
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="name"\r\n\r\n{name}\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{Path(wav_path).name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + data + f"\r\n--{boundary}--\r\n".encode("utf-8")
    voice = _soniox_api(key, "POST", "/v1/voices", body, f"multipart/form-data; boundary={boundary}")
    deadline = time.monotonic() + timeout_s
    while True:
        status = _tts_status(voice)
        if status == "ready":
            return voice["id"]
        if status.startswith("failed"):
            raise RuntimeError(f"Soniox couldn't make the voice: {status}")
        if time.monotonic() > deadline:
            raise RuntimeError("Soniox is still processing the voice — check again later (Refresh).")
        time.sleep(2.0)
        voice = _soniox_api(key, "GET", f"/v1/voices/{voice['id']}")


def soniox_delete_voice(key, voice_id):
    _soniox_api(key, "DELETE", f"/v1/voices/{voice_id}")


def voices_for(engine, lang):
    if engine == "soniox":
        return SONIOX_VOICES.get(lang, SONIOX_VOICES_OTHER)
    table = AZURE_VOICES if engine == "azure" else PIPER_VOICES
    return table.get(lang, ())


@dataclass
class VoiceSettings:
    engine: str = "off"  # "off" | "local" | "azure" | "soniox"
    voice: str = ""
    # Language the voice speaks (may differ from the captions'). When the
    # recognizer didn't already provide it, the source text is translated
    # offline; `source_langs` are the spoken languages to preload for that.
    lang: str = "pt"
    source_langs: tuple = ()
    # Its sentences come from a recognition session of its own (a second
    # Soniox connection translating into its language): it speaks only
    # those, marked info["voice_only"] = its language.
    own_stream: bool = False
    label: str = "Voice"  # for the Log ("Voice 2 error: ...")
    device_index: int = None  # audio output device, or None
    device_channels: tuple = (0, 1)  # the device's channels it plays on (0-based)
    # Channels of the NDI output's audio it plays on (0-based; () = not
    # sent), and/or a separate, audio-only NDI source with this name.
    ndi_channels: tuple = ()
    ndi_source: str = ""
    azure_key: str = ""
    azure_region: str = ""
    soniox_key: str = ""


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


def _approach(start, target, n, tau):
    """n samples of a one-pole glide from `start` towards `target`."""
    return target + (start - target) * np.exp(-np.arange(1, n + 1) / tau)


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


class _StreamResampler:
    """Resamples one sentence that arrives in chunks (libswresample via
    PyAV, kept open so the chunks join without clicks), with a one-shot
    fallback at the end."""

    def __init__(self, in_rate, out_rate):
        self.in_rate, self.out_rate = in_rate, out_rate
        self._av = None
        self._held = []
        if in_rate != out_rate:
            try:
                import av

                self._av = av
                self._resampler = av.AudioResampler(format="flt", layout="mono", rate=out_rate)
            except Exception:
                self._av = None

    def process(self, x):
        if self.in_rate == self.out_rate:
            return x
        if self._av is None:
            self._held.append(x)
            return np.zeros(0, np.float32)
        frame = self._av.AudioFrame.from_ndarray(x.astype(np.float32)[None, :], format="flt", layout="mono")
        frame.sample_rate = self.in_rate
        return self._join(self._resampler.resample(frame))

    def flush(self):
        if self.in_rate == self.out_rate:
            return np.zeros(0, np.float32)
        if self._av is None:
            return _resample(np.concatenate(self._held), self.in_rate, self.out_rate) if self._held else np.zeros(0, np.float32)
        return self._join(self._resampler.resample(None))

    @staticmethod
    def _join(frames):
        if not frames:
            return np.zeros(0, np.float32)
        return np.concatenate([f.to_ndarray().reshape(-1) for f in frames]).astype(np.float32)


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


class _SonioxSynth:
    """Soniox text-to-speech over its REST API: one request per sentence,
    raw 16-bit PCM back."""

    rate = 24000

    def __init__(self, name, lang, key):
        if not key.strip():
            raise RuntimeError("Soniox voices need the Soniox key (Input panel).")
        self._name = name
        self._lang = lang
        self._key = key.strip()

    def stream(self, text, speed):
        """Yields the audio as it arrives: the first ~0.1 s after ~0.5 s,
        then faster than real time — the voice starts playing at once
        instead of after the whole sentence (~4 s for a long one)."""
        import json
        import urllib.error

        body = {
            "model": SONIOX_TTS_MODEL, "language": self._lang, "voice": self._name,
            "audio_format": "pcm_s16le", "sample_rate": self.rate, "text": text,
            "speed": min(SONIOX_SPEED_RANGE[1], max(SONIOX_SPEED_RANGE[0], speed)),
        }
        request = urllib.request.Request(
            SONIOX_TTS_URL, data=json.dumps(body).encode("utf-8"), method="POST",
            # Soniox's firewall rejects urllib's default User-Agent (403).
            headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {self._key}",
                "User-Agent": "AlphaSubs",
            },
        )
        try:
            resp = urllib.request.urlopen(request, timeout=15)
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read()).get("error_message", "")
            except Exception:
                detail = ""
            raise RuntimeError(f"Soniox TTS error {e.code}: {detail or e.reason}")
        with resp:
            odd = b""
            while True:
                data = resp.read1(SONIOX_CHUNK_BYTES)
                if not data:
                    return
                data = odd + data
                cut = len(data) // 2 * 2
                data, odd = data[:cut], data[cut:]
                yield np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


class _AudioBuffer:
    """FIFO of mono float32 samples for one output, at that output's rate."""

    def __init__(self, rate):
        self.rate = rate
        self.active = True  # False: appends are dropped (a monitor that is off)
        self._lock = threading.Lock()
        self._chunks = []
        self._len = 0

    def append(self, samples):
        if not self.active:
            return
        with self._lock:
            self._chunks.append(samples)
            self._len += len(samples)

    def clear(self):
        with self._lock:
            self._chunks, self._len = [], 0

    def seconds(self):
        return self._len / self.rate

    def read(self, n):
        return self.read_count(n)[0]

    def read_count(self, n):
        """(n samples, how many of them were audio rather than padding)."""
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
        return out, filled


class _LiveBuffer(_AudioBuffer):
    """The original sound, fed live from the input on its own clock: reads
    wait for a little slack to build up (again after running dry), and a
    backlog past ORIGINAL_MAX_S is dropped so it can't add delay."""

    def __init__(self, rate):
        super().__init__(rate)
        self._primed = False
        self._prefill = int(rate * ORIGINAL_PREFILL_S)

    def append(self, samples):
        super().append(samples)
        with self._lock:
            limit = self._prefill + int(self.rate * ORIGINAL_MAX_S)
            while self._len > limit and len(self._chunks) > 1:
                self._len -= len(self._chunks.pop(0))

    def read(self, n):
        if not self._primed:
            if self._len < n + self._prefill:
                return np.zeros(n, dtype=np.float32)
            self._primed = True
        out, filled = self.read_count(n)
        if filled < n:
            self._primed = False
        return out


class _Route:
    """One voice on one output: its buffer and channels (or a function
    returning them, for the monitor), and the ducking state it applies to
    the original on those channels."""

    def __init__(self, buf, channels, gain):
        self.buf, self._channels, self.gain = buf, channels, gain
        self.env = 1.0  # the original's current gain under this voice
        self.hold = 0  # samples the original stays down after the voice

    @property
    def channels(self):
        return self._channels() if callable(self._channels) else self._channels

    def duck(self, filled, n, rate, level):
        """The original's gain for the next n samples, the voice having
        played the first `filled` of them."""
        if filled:
            self.hold = int(rate * DUCK_HOLD_S)
            down = n
        else:
            down = min(n, self.hold)
        self.hold = max(0, self.hold - (n - filled))
        parts = []
        if down:
            parts.append(_approach(self.env, level, down, rate * DUCK_ATTACK_S))
            self.env = float(parts[-1][-1])
        if n - down:
            parts.append(_approach(self.env, 1.0, n - down, rate * DUCK_RELEASE_S))
            self.env = float(parts[-1][-1])
        return np.concatenate(parts).astype(np.float32)


class _Mix:
    """One audio output (a sound device or an NDI source), shared by the
    voices routed to it: each voice has its own buffer, played on its own
    channels, so two voices don't queue behind each other. With
    `original_gain` set (a function returning the original's level under
    the voice, or None for none), the original plays on the voices'
    channels too, lowered while the voice on that channel speaks."""

    def __init__(self, rate, channels):
        self.rate = rate
        self.channels = channels
        self.original = _LiveBuffer(rate)
        self.original_gain = lambda: None
        self._routes = []
        self._resampler = None

    def route(self, channels, gain):
        buf = _AudioBuffer(self.rate)
        self._routes.append(_Route(buf, channels, gain))
        return buf

    def feed_original(self, rate, mono):
        """Called from the input's thread (a sound device callback: keep it
        light — numpy only)."""
        if self.original_gain() is None:
            if self.original.seconds():
                self.original.clear()
            return
        if self._resampler is None or self._resampler_rate != rate:
            from audio_io import _Resampler

            self._resampler, self._resampler_rate = _Resampler(rate, self.rate), rate
        self.original.append(self._resampler.process(np.array(mono, dtype=np.float32, copy=True)))

    def read(self, n):
        """(channels, n) float32."""
        out = np.zeros((self.channels, n), dtype=np.float32)
        level = self.original_gain()
        envs = {}  # channel -> the original's gain there (the most ducked voice wins)
        for r in self._routes:
            if not r.buf.active:
                continue
            block, filled = r.buf.read_count(n)
            gain = r.gain()
            channels = [c for c in r.channels if c < self.channels]
            for c in channels:
                out[c] += block * gain
            if level is not None:
                env = r.duck(filled if gain > 0 else 0, n, self.rate, level)
                for c in channels:
                    envs[c] = np.minimum(envs[c], env) if c in envs else env
        if level is not None and envs:
            original = self.original.read(n)
            for c, env in envs.items():
                out[c] += original * env
        return np.clip(out, -1.0, 1.0)


class _DeviceOutput:
    def __init__(self, device_index, channels, buffer_s, mix=None):
        """`device_index` None: the system's default output. `mix`: an
        existing _Mix at the device's rate (the monitor's)."""
        import sounddevice as sd

        info = sd.query_devices(device_index, kind="output")
        self.name = info["name"]
        self.mix = mix or _Mix(int(info["default_samplerate"]), channels)
        self._stream = sd.OutputStream(
            device=device_index, samplerate=self.mix.rate, channels=self.mix.channels, dtype="float32",
            latency=buffer_s, callback=self._callback,
        )

    def start(self):
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()

    def _callback(self, outdata, frames, _time_info, _status):
        outdata[:] = self.mix.read(frames).T


class _NdiOutput:
    """Sends a continuous 48 kHz stream (silence between sentences) in
    frames of the buffer size, paced on an absolute schedule (not by SDK
    clocking, which would freeze the app — see ndi_io.Sender). The SDK's
    synthesized timecodes follow the sample count, so receivers still see an
    even stream. `own`: the sender is this output's (closed on stop)."""

    def __init__(self, sender, channels, buffer_s, log, own=False):
        self.mix = _Mix(NDI_RATE, channels)
        self._sender = sender
        self._own = own
        self._n = int(NDI_RATE * buffer_s)
        self.log = log
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3.0)
        if self._own:
            self._sender.close()

    def _loop(self):
        n = self._n
        next_t = time.perf_counter()
        while not self._stop.is_set():
            try:
                self._sender.send_audio(self.mix.read(n), NDI_RATE)
            except Exception as e:
                self.log(f"NDI audio error: {e}")
                return
            next_t += n / NDI_RATE
            delay = next_t - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
            elif delay < -0.5:
                next_t = time.perf_counter()


class _Voice:
    """One voice: synthesizes each sentence in its language and queues it on
    the outputs it is attached to."""

    def __init__(self, settings, tuning, log):
        self.settings = settings
        self.tuning = tuning
        self.log = log
        self.label = settings.label
        self._texts = queue.Queue()
        self._buffers = []
        self.monitor = None  # its buffer on the monitor output (see VoiceOutput.set_monitor)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._synth_loop, daemon=True)

    def attach(self, mix, channels):
        self._buffers.append(mix.route(channels, self._gain))

    def _targets(self):
        """The buffers it plays into now (the monitor's only while it's on)."""
        monitor = self.monitor
        return self._buffers + [monitor] if monitor is not None and monitor.active else self._buffers

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3.0)

    def say(self, info):
        if info.get("lang") == self.settings.lang:
            return  # said in its own language: the listeners heard the original
        only = info.get("voice_only")
        if self.settings.own_stream and only == self.settings.lang or not self.settings.own_stream and not only:
            self._texts.put((time.monotonic(), info))

    def _pending_s(self):
        audio = max((b.seconds() for b in self._targets()), default=0.0)
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
            elif s.engine == "soniox":
                synth = _SonioxSynth(s.voice, s.lang, s.soniox_key)
            else:
                synth = _PiperSynth(s.voice, self.log)
            for src in s.source_langs:
                if src != s.lang:
                    self._translator(src)  # load (and download) now, not on the first sentence
            self.log(f"{self.label} ready ({s.voice}, speaking {s.lang}).")
        except Exception as e:
            self.log(f"{self.label} error: {e}")
            return
        while not self._stop.is_set():
            try:
                queued_at, info = self._texts.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.tuning.muted:
                continue
            waited = time.monotonic() - queued_at
            buffered = max((b.seconds() for b in self._targets()), default=0.0)
            if waited + buffered > MAX_LAG_S:
                self.log(f"{self.label} too far behind — skipped: {info.get('source', '')[:60]}")
                continue
            backlog = max(0.0, self._pending_s() - SPEEDUP_START_S)
            speed = self.tuning.speed * min(SPEEDUP_MAX, 1.0 + backlog * SPEEDUP_PER_S)
            try:
                text = self._text_for(info)
                if not text:
                    continue
                if hasattr(synth, "stream"):
                    self._play(synth, synth.stream(text, speed))
                else:
                    pcm = synth.synth(text, speed)
                    for buf in self._targets():
                        buf.append(_resample(pcm, synth.rate, buf.rate))
            except Exception as e:
                self.log(f"{self.label} error: {e}")

    def _play(self, synth, chunks):
        """Queues a sentence on the outputs chunk by chunk, as it's
        synthesized."""
        resamplers = [(buf, _StreamResampler(synth.rate, buf.rate)) for buf in self._targets()]
        try:
            for pcm in chunks:
                if self._stop.is_set() or self.tuning.muted:
                    return
                for buf, resampler in resamplers:
                    buf.append(resampler.process(pcm))
        finally:
            for buf, resampler in resamplers:
                buf.append(resampler.flush())

    def _gain(self):
        return 0.0 if self.tuning.muted else self.tuning.volume


class VoiceOutput:
    """The translated voices — one or two, each with its own VoiceSettings
    and VoiceTuning — and the outputs they play on. Voices on the same sound
    device or on the NDI output's audio share it, each on its own channels.
    The buffer size is the first voice's."""

    def __init__(self, voices, ndi_sender, make_sender, log, duck_db=None):
        """`voices`: [(VoiceSettings, VoiceTuning)]. `ndi_sender`: the NDI
        output (for voices with ndi_channels). `make_sender(name)`: creates
        an NDI sender, for voices with an ndi_source of their own.
        `duck_db`: see the attribute."""
        self.log = log
        self._voices = [_Voice(s, t, log) for s, t in voices]
        self._buffer_s = max(0.02, voices[0][1].buffer_ms / 1000.0)
        self._ndi_sender = ndi_sender
        self._make_sender = make_sender
        self._outputs = []
        self._monitor_mix = None
        self._monitor_out = None
        self._monitored = set()
        # The original's level (dB) under a speaking voice, on the voices'
        # channels (None: the voices only). Settable live.
        self.duck_db = duck_db

    def _original_gain(self):
        if self.duck_db is None or all(v.tuning.muted for v in self._voices):
            return None
        return 10.0 ** (self.duck_db / 20.0)

    def start(self):
        try:
            self._open_outputs()
            for out in self._outputs:
                out.start()
        except Exception:
            self._close_outputs()
            raise
        for out in self._outputs:
            out.mix.original_gain = self._original_gain
        self._prepare_monitor()
        for v in self._voices:
            v.start()

    def _prepare_monitor(self):
        """Routes every voice to a mix for the default output, inactive
        until set_monitor() turns it on (the device opens only then)."""
        try:
            import sounddevice as sd

            info = sd.query_devices(kind="output")
            mix = _Mix(int(info["default_samplerate"]), min(2, int(info["max_output_channels"])))
        except Exception as e:
            self.log(f"No default audio output to monitor the voices: {e}")
            return
        mix.original_gain = self._original_gain
        for v in self._voices:
            v.monitor = mix.route(lambda v=v: self._monitor_channels(v), v._gain)
            v.monitor.active = False
        self._monitor_mix = mix

    def _monitor_channels(self, v):
        """Alone: both sides. Two voices monitored: first left, second right."""
        if self._monitor_mix.channels < 2:
            return (0,)
        if len(self._monitored) > 1:
            return (self._voices.index(v) % 2,)
        return (0, 1)

    def set_monitor(self, labels):
        """Plays the voices whose label is in `labels` (with the original
        under them, as on the outputs) on the computer's default output;
        none closes it. Returns the output's name, or None. Called from the
        UI thread; raises if the device can't be opened."""
        mix = self._monitor_mix
        if mix is None:
            return None
        self._monitored = {v.label for v in self._voices if v.label in labels}
        for v in self._voices:
            on = v.label in self._monitored
            if not on:
                v.monitor.clear()
            v.monitor.active = on
        if self._monitored and self._monitor_out is None:
            out = _DeviceOutput(None, mix.channels, self._buffer_s, mix=mix)
            out.start()
            self._monitor_out = out
        elif not self._monitored and self._monitor_out is not None:
            self._close_monitor()
        return self._monitor_out.name if self._monitor_out is not None else None

    def _close_monitor(self):
        out, self._monitor_out = self._monitor_out, None
        if out is not None:
            try:
                out.stop()
            except Exception:
                pass
            self._monitor_mix.original.clear()  # not stale when it reopens

    def feed_original(self, rate, mono):
        """The input's audio, live (the speech engine's capture calls this
        from its thread): the original under the voices."""
        for out in self._outputs:
            out.mix.feed_original(rate, mono)
        if self._monitor_out is not None:
            self._monitor_mix.feed_original(rate, mono)

    def _open_outputs(self):
        voices = self._voices
        devices = {}
        for v in voices:
            if v.settings.device_index is not None:
                devices.setdefault(v.settings.device_index, []).append(v)
        for index, members in devices.items():
            import sounddevice as sd

            available = int(sd.query_devices(index)["max_output_channels"])
            routes = []
            for v in members:
                channels = tuple(c for c in v.settings.device_channels if c < available) or (0,)
                if channels != tuple(v.settings.device_channels):
                    self.log(f"{v.label}: the device has {available} channel(s) — playing on channel {channels[0] + 1}.")
                routes.append((v, channels))
            count = min(available, max(2, max(c for _v, chans in routes for c in chans) + 1))
            out = _DeviceOutput(index, count, self._buffer_s)
            self._outputs.append(out)
            for v, channels in routes:
                v.attach(out.mix, channels)
        on_ndi = [v for v in voices if v.settings.ndi_channels]
        if on_ndi and self._ndi_sender is not None:
            count = max(2, max(c for v in on_ndi for c in v.settings.ndi_channels) + 1)
            out = _NdiOutput(self._ndi_sender, count, self._buffer_s, self.log)
            self._outputs.append(out)
            for v in on_ndi:
                v.attach(out.mix, v.settings.ndi_channels)
        for v in voices:
            if v.settings.ndi_source:
                sender = self._make_sender(v.settings.ndi_source)
                out = _NdiOutput(sender, 2, self._buffer_s, self.log, own=True)
                self._outputs.append(out)
                v.attach(out.mix, (0, 1))
                self.log(f"{v.label}: NDI source '{v.settings.ndi_source}' (audio only).")

    def _close_outputs(self):
        for out in self._outputs:
            try:
                out.stop()
            except Exception:
                pass
        self._outputs = []

    def stop(self):
        for v in self._voices:
            v._stop.set()
        for v in self._voices:
            v.stop()
        self._close_outputs()
        self._close_monitor()

    def say(self, info):
        """Queues one final sentence: the `info` dict from the speech
        engine (see speech.py), so each voice can speak its own language."""
        for v in self._voices:
            v.say(info)
