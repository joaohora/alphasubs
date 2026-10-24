"""Speech-to-text (+ translation) engines for the audio mode.

Each engine captures one audio input device, recognizes EN/PT/ES speech and
reports caption text in the target language through
on_caption(text, final): partial (still-changing) results as the speaker
talks, then a final one per utterance.

- LocalWhisperEngine: faster-whisper (GPU via CUDA when available, CPU
  otherwise) + offline Argos/CTranslate2 translation. Runs fully offline
  once the models are downloaded.
- AzureEngine: Azure AI Speech translation (cloud, streaming, lower
  latency; needs a key/region and an internet connection).

Heavy imports (faster_whisper, azure SDK) happen inside the engine thread so
the app starts fast and NDI-only use doesn't need them."""

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import audio_io

SR = audio_io.SAMPLE_RATE
LANGS = ("en", "pt", "es")
AZURE_LOCALES = {"en": "en-US", "pt": "pt-PT", "es": "es-ES"}
AZURE_TARGETS = {"pt": "pt-pt", "en": "en"}
WHISPER_DIR = Path.home() / ".alphasubs_models" / "whisper"

# Streaming segmentation (local engine). Every STEP_S the current utterance
# is re-transcribed (partial caption); an utterance is finalized after
# END_SILENCE_S of silence, or force-split at MAX_UTTERANCE_S so long
# unbroken speech still produces captions within the latency budget.
STEP_S = 0.8
MIN_AUDIO_S = 0.5
END_SILENCE_S = 0.6
MAX_UTTERANCE_S = 8.0
LEAD_IN_S = 0.3
LANG_LOCK_S = 2.5
PROMPT_CHARS = 200
# Credits Whisper is known to "hear" in silence/noise (from its training
# subtitles) — dropped whenever they show up in a result.
HALLUCINATION_MARKERS = (
    "amara.org", "legendas pela comunidade", "subtítulos realizados por",
    "subtitles by", "thanks for watching", "obrigado por assistir", "inscreva-se",
)


@dataclass
class SpeechSettings:
    engine: str = "local"  # "local" | "azure"
    device_index: int = None
    channel: int = None  # 0-based, or None = mix of the first two channels
    source_lang: str = "auto"  # "auto" | one of LANGS
    target_lang: str = "pt"  # "pt" | "en"
    whisper_model: str = "auto"
    # Comma-separated names/terms to bias recognition towards.
    hotwords: str = ""
    azure_key: str = ""
    azure_region: str = ""


def create_engine(settings, on_caption, log, on_status):
    cls = AzureEngine if settings.engine == "azure" else LocalWhisperEngine
    return cls(settings, on_caption, log, on_status)


def _add_windows_cuda_dirs():
    """CTranslate2 needs cuBLAS/cuDNN DLLs on Windows. Makes the ones from
    the nvidia-* pip packages, or a 'cuda' folder next to the app/exe,
    loadable (a system CUDA install on PATH works without this)."""
    if sys.platform != "win32":
        return
    candidates = [Path(sys.executable).parent / "cuda", Path(__file__).resolve().parent.parent / "cuda"]
    try:
        import nvidia
        for base in nvidia.__path__:
            candidates.extend(p for p in Path(base).glob("*/bin") if p.is_dir())
    except ImportError:
        pass
    for path in candidates:
        if path.is_dir():
            os.add_dll_directory(str(path))
            os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")


class _Engine:
    def __init__(self, settings, on_caption, log, on_status):
        self.settings = settings
        self.on_caption = on_caption
        self.log = log
        self.on_status = on_status
        self.error = None
        self.capture = None
        self.stop_event = threading.Event()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)

    @property
    def level_db(self):
        return self.capture.level_db if self.capture is not None else -120.0

    def start(self):
        self._thread.start()

    def stop(self, timeout=3.0):
        self.stop_event.set()
        self._thread.join(timeout=timeout)

    def _open_capture(self):
        s = self.settings
        if s.device_index is None:
            raise RuntimeError("No audio input device selected.")
        self.capture = audio_io.AudioCapture(s.device_index, s.channel)
        self.capture.start()

    def _thread_main(self):
        try:
            self._run()
        except Exception as e:
            self.error = str(e)
            self.log(f"Speech engine error: {e}")
            self.on_status("Speech engine error — see Log.")
        finally:
            if self.capture is not None:
                self.capture.close()

    def _run(self):
        raise NotImplementedError


class LocalWhisperEngine(_Engine):
    def _run(self):
        _add_windows_cuda_dirs()
        import ctranslate2
        from faster_whisper import WhisperModel
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        from translator import Translator

        s = self.settings
        self.on_status("Loading speech model...")
        self.model, device = self._load_model(WhisperModel, ctranslate2)
        if self.stop_event.is_set():
            return

        sources = LANGS if s.source_lang == "auto" else (s.source_lang,)
        self.translators = {}
        for src in sources:
            if src != s.target_lang:
                self.on_status(f"Loading translation {src} -> {s.target_lang}...")
                self.translators[src] = Translator(src, s.target_lang, log=self.log)
            if self.stop_event.is_set():
                return

        self._open_capture()
        self.log(f"Listening (Whisper on {device.upper()}, {s.source_lang} -> {s.target_lang}).")
        self.on_status("Listening.")

        vad_opts = VadOptions(threshold=0.5, min_silence_duration_ms=250, speech_pad_ms=100)
        self._prompt = ("", "")  # (language, last final text) — context for the next utterance
        self._utt_lang = None
        buf = np.zeros(0, dtype=np.float32)
        showing_partial = False
        last_step = time.monotonic()

        while not self.stop_event.is_set():
            block = self.capture.read(timeout=0.1)
            if block is not None and len(block):
                buf = np.concatenate([buf, block])
            now = time.monotonic()
            if now - last_step < STEP_S or len(buf) < MIN_AUDIO_S * SR:
                continue
            last_step = now

            speech = get_speech_timestamps(buf, vad_opts)
            if not speech:
                buf = buf[-int(LEAD_IN_S * SR):]
                if showing_partial:
                    self.on_caption("", True)  # false start: drop the partial
                    showing_partial = False
                self._utt_lang = None
                continue

            start = max(0, speech[0]["start"] - int(LEAD_IN_S * SR))
            trailing_s = (len(buf) - speech[-1]["end"]) / SR
            if trailing_s >= END_SILENCE_S:
                cut = min(len(buf), speech[-1]["end"] + int(0.1 * SR))
            elif (len(buf) - start) / SR >= MAX_UTTERANCE_S:
                cut = self._forced_cut(buf, speech, start)
            else:
                text = self._transcribe(buf[start:], final=False)
                self.on_caption(text, False)
                showing_partial = True
                continue

            text = self._transcribe(buf[start:cut], final=True)
            self.on_caption(text, True)
            showing_partial = False
            buf = buf[cut:]
            self._utt_lang = None

    def _load_model(self, WhisperModel, ctranslate2):
        s = self.settings
        has_cuda = ctranslate2.get_cuda_device_count() > 0
        attempts = ([("cuda", "float16")] if has_cuda else []) + [("cpu", "int8")]
        for device, compute_type in attempts:
            name = s.whisper_model
            if name == "auto":
                name = "large-v3-turbo" if device == "cuda" else "small"
            try:
                self.log(f"Loading Whisper '{name}' on {device.upper()} (the first run downloads it)...")
                model = WhisperModel(
                    name, device=device, compute_type=compute_type,
                    cpu_threads=min(8, os.cpu_count() or 4), download_root=str(WHISPER_DIR),
                )
                # Warm-up: surfaces missing CUDA/cuDNN libraries now rather
                # than on the first spoken word.
                segments, _ = model.transcribe(
                    np.zeros(SR, dtype=np.float32), language="en", beam_size=1, without_timestamps=True
                )
                list(segments)
                return model, device
            except Exception as e:
                if device == "cpu":
                    raise
                self.log(f"GPU unavailable ({e}) — falling back to CPU.")
        raise RuntimeError("Could not load the Whisper model.")

    def _forced_cut(self, buf, speech, start):
        """Where to split an utterance that hit MAX_UTTERANCE_S: at the
        latest pause between VAD speech segments if there is one, otherwise
        at the quietest 100 ms of the last few seconds (least likely to cut a
        word in half)."""
        min_cut = start + 2 * SR
        gaps = [seg["end"] for seg in speech[:-1] if seg["end"] > min_cut]
        if gaps:
            return gaps[-1]
        win = int(0.1 * SR)
        lo = max(min_cut, len(buf) - 3 * SR)
        hi = len(buf) - win
        if hi <= lo:
            return len(buf)
        frames = buf[lo:hi - (hi - lo) % win].reshape(-1, win)
        quietest = int(np.argmin(np.mean(frames * frames, axis=1)))
        return lo + quietest * win + win // 2

    def _language(self, audio):
        s = self.settings
        if s.source_lang != "auto":
            return s.source_lang
        if self._utt_lang and len(audio) >= LANG_LOCK_S * SR:
            return self._utt_lang
        _lang, _prob, probs = self.model.detect_language(audio)
        allowed = [(lang, p) for lang, p in probs if lang in LANGS]
        self._utt_lang = max(allowed, key=lambda lp: lp[1])[0] if allowed else "en"
        return self._utt_lang

    def _transcribe(self, audio, final):
        s = self.settings
        lang = self._language(audio)
        prompt_lang, prompt_text = self._prompt
        segments, _info = self.model.transcribe(
            audio,
            language=lang,
            task="transcribe",
            beam_size=5 if final else 1,
            temperature=[0.0, 0.2, 0.4] if final else 0.0,
            condition_on_previous_text=False,
            initial_prompt=prompt_text if prompt_lang == lang and prompt_text else None,
            hotwords=s.hotwords.strip() or None,
            without_timestamps=True,
        )
        parts = [
            seg.text.strip() for seg in segments
            if not (seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0)
        ]
        text = " ".join(p for p in parts if p)
        if any(marker in text.lower() for marker in HALLUCINATION_MARKERS):
            text = ""
        if final and text:
            self._prompt = (lang, text[-PROMPT_CHARS:])
        if text and lang in self.translators:
            text = self.translators[lang].translate(text)
        return text


class AzureEngine(_Engine):
    def _run(self):
        import azure.cognitiveservices.speech as speechsdk

        s = self.settings
        key, region = s.azure_key.strip(), s.azure_region.strip()
        if not key or not region:
            raise RuntimeError("Azure Speech needs a key and a region (Input panel).")

        if s.source_lang == "auto":
            # Continuous language identification requires the v2 endpoint.
            config = speechsdk.translation.SpeechTranslationConfig(
                subscription=key,
                endpoint=f"wss://{region}.stt.speech.microsoft.com/speech/universal/v2",
            )
            config.set_property(speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, "Continuous")
            auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                languages=[AZURE_LOCALES[lang] for lang in LANGS]
            )
        else:
            config = speechsdk.translation.SpeechTranslationConfig(subscription=key, region=region)
            config.speech_recognition_language = AZURE_LOCALES[s.source_lang]
            auto_detect = None
        config.add_target_language(AZURE_TARGETS[s.target_lang])

        stream = speechsdk.audio.PushAudioInputStream(
            stream_format=speechsdk.audio.AudioStreamFormat(samples_per_second=SR, bits_per_sample=16, channels=1)
        )
        recognizer = speechsdk.translation.TranslationRecognizer(
            translation_config=config,
            auto_detect_source_language_config=auto_detect,
            audio_config=speechsdk.audio.AudioConfig(stream=stream),
        )
        phrases = [p.strip() for p in s.hotwords.split(",") if p.strip()]
        if phrases:
            grammar = speechsdk.PhraseListGrammar.from_recognizer(recognizer)
            for phrase in phrases:
                grammar.addPhrase(phrase)

        def caption_text(result):
            detected = result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
            ) or AZURE_LOCALES.get(s.source_lang, "")
            if detected.lower().startswith(s.target_lang):
                return result.text  # already in the target language
            return next(iter(result.translations.values()), "") if result.translations else ""

        def on_recognizing(evt):
            self.on_caption(caption_text(evt.result), False)

        def on_recognized(evt):
            if evt.result.reason == speechsdk.ResultReason.TranslatedSpeech:
                self.on_caption(caption_text(evt.result), True)

        def on_canceled(evt):
            if evt.reason == speechsdk.CancellationReason.Error:
                self.error = evt.error_details
                self.log(f"Azure Speech error: {evt.error_details}")
                self.on_status("Speech engine error — see Log.")

        recognizer.recognizing.connect(on_recognizing)
        recognizer.recognized.connect(on_recognized)
        recognizer.canceled.connect(on_canceled)

        self._open_capture()
        recognizer.start_continuous_recognition()
        self.log(f"Listening (Azure Speech, region {region}, {s.source_lang} -> {s.target_lang}).")
        self.on_status("Listening.")
        try:
            while not self.stop_event.is_set():
                block = self.capture.read(timeout=0.1)
                if block is not None and len(block):
                    stream.write((np.clip(block, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
        finally:
            stream.close()
            recognizer.stop_continuous_recognition()
