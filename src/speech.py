"""Speech-to-text (+ translation) engines for the audio mode.

Each engine captures one audio input device, recognizes EN/PT/ES speech and
reports caption text in the target language through
on_caption(text, final, info): partial (still-changing) results as the
speaker talks, then a final one per utterance. Finals carry `info` =
{"lang": source language, "source": source-language text, "texts":
{language: text}} so other consumers (the translated voice) can use a
different language than the captions.

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
AZURE_SEGMENT_STRATEGY = "Semantic"
# A final of at most this many words (a fragment at a hesitation: "No final."
# + "haverá uma conversa...") is held while the person keeps talking, and
# joined to the next final. It goes out alone once Azure has reported no
# speech for AZURE_MERGE_IDLE_S, or after AZURE_MERGE_MAX_S at most. Longer
# finals go out at once.
AZURE_SHORT_WORDS = 5
AZURE_MERGE_IDLE_S = 1.6
AZURE_MERGE_MAX_S = 8.0
WHISPER_DIR = Path.home() / ".alphasubs_models" / "whisper"
# "auto" on an NVIDIA GPU: the best model that loads (smaller GPUs can't fit
# large-v3-turbo), in the fastest compute type the GPU supports.
CUDA_AUTO_MODELS = ("large-v3-turbo", "small", "base")
CUDA_COMPUTE_TYPES = ("float16", "int8_float16", "int8", "float32")

# Streaming segmentation (local engine). Every STEP_S the current utterance
# is re-transcribed (partial caption); an utterance is finalized after
# END_SILENCE_S of silence, or force-split at MAX_UTTERANCE_S so long
# unbroken speech still produces captions within the latency budget.
STEP_S = 0.8
MIN_AUDIO_S = 0.5
END_SILENCE_S = 0.5
MAX_UTTERANCE_S = 8.0
LEAD_IN_S = 0.3
LANG_DETECT_S = 2.0
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
    # An NDI source to take the audio from instead of a sound device.
    ndi_source: object = None
    channel: int = None  # 0-based, or None = mix of the first two channels
    source_lang: str = "auto"  # "auto" | one of LANGS
    target_lang: str = "pt"  # "pt" | "en"
    whisper_model: str = "auto"
    # Comma-separated names/terms to bias recognition towards.
    hotwords: str = ""
    azure_key: str = ""
    azure_region: str = ""
    # Further languages to translate finals into, besides target_lang (the
    # voice's language). Only Azure uses it — it translates to several
    # languages at once; local consumers translate `info["source"]` instead.
    extra_target_langs: tuple = ()


def _best_cuda_compute_type(ctranslate2):
    try:
        supported = ctranslate2.get_supported_compute_types("cuda")
    except Exception:
        return "float16"
    return next((t for t in CUDA_COMPUTE_TYPES if t in supported), "float32")


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
        self.listener = None
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

    def set_listener(self, listener):
        """Also plays the captured audio on `listener` (an
        audio_io.ListenOutput), or stops with None. Callable from any thread,
        before or after the capture opens (that waits for the models)."""
        self.listener = listener
        capture = self.capture
        if capture is not None:
            capture.listener = listener

    def _open_capture(self):
        s = self.settings
        if s.ndi_source is not None:
            self.capture = audio_io.NDIAudioCapture(s.ndi_source, s.channel)
        elif s.device_index is None:
            raise RuntimeError("No audio input device selected.")
        else:
            self.capture = audio_io.AudioCapture(s.device_index, s.channel)
        self.capture.listener = self.listener
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

        self._final_beam = 5 if device.startswith("cuda") else 2
        self._last_source = ("", "")
        self._pass_s = None  # moving average of one transcription pass (s)
        self._open_capture()
        self.log(f"Listening (Whisper on {device}, {s.source_lang} -> {s.target_lang}).")
        self.on_status("Listening.")

        vad_opts = VadOptions(threshold=0.5, min_silence_duration_ms=250, speech_pad_ms=100)
        self._prompt = ("", "")  # (language, last final text) — context for the next utterance
        self._utt_lang = None  # language detected for the current utterance
        self._last_lang = None  # language of the previous utterance
        buf = np.zeros(0, dtype=np.float32)
        showing_partial = False
        # The last partial: (buf index it covered up to, caption text) — if it
        # already heard the whole utterance, it becomes the final as is,
        # saving a full pass at the end of each sentence.
        last_partial = None
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
                last_partial = None
                continue

            start = max(0, speech[0]["start"] - int(LEAD_IN_S * SR))
            # An utterance ends at the first pause of END_SILENCE_S — whether
            # it's at the end of the buffer or already followed by new speech
            # (when a pass is slow, the next sentence can start before we look
            # again; it must not be merged into this one).
            pause_at = next(
                (i for i in range(len(speech) - 1)
                 if speech[i + 1]["start"] - speech[i]["end"] >= END_SILENCE_S * SR),
                None,
            )
            trailing_s = (len(buf) - speech[-1]["end"]) / SR
            if pause_at is not None:
                speech = speech[:pause_at + 1]
                cut = speech[-1]["end"] + int(0.1 * SR)
            elif trailing_s >= END_SILENCE_S:
                cut = min(len(buf), speech[-1]["end"] + int(0.1 * SR))
            elif (len(buf) - start) / SR >= MAX_UTTERANCE_S:
                cut = self._forced_cut(buf, speech, start)
            else:
                covered = len(buf)
                text = self._transcribe(buf[start:], final=False)
                if text is not None:
                    self.on_caption(text, False)
                    showing_partial = True
                    last_partial = (covered, text, self._last_source)
                continue

            if last_partial and last_partial[0] >= speech[-1]["end"] and last_partial[0] <= cut + END_SILENCE_S * SR:
                text, source = last_partial[1], last_partial[2]
                self._remember_prompt(*source)
            else:
                text = self._transcribe(buf[start:cut], final=True)
                source = self._last_source
            self.on_caption(text, True, {"lang": source[0], "source": source[1], "texts": {s.target_lang: text}})
            showing_partial = False
            last_partial = None
            buf = buf[cut:]
            self._utt_lang = None

    def _load_model(self, WhisperModel, ctranslate2):
        s = self.settings
        try:
            has_cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception as e:
            has_cuda = False
            self.log(f"CUDA check failed: {e}")
        if not has_cuda and sys.platform == "win32":
            self.log(
                "No NVIDIA GPU visible to CUDA — using the CPU. Check that the NVIDIA driver is "
                "installed and recent ('nvidia-smi' in a terminal should list the GPU)."
            )
        auto = s.whisper_model == "auto"
        attempts = []
        if has_cuda:
            # Older GPUs (e.g. Maxwell laptop chips) lack float16/int8
            # support and have little memory: use the best compute type the
            # GPU supports, and with "auto" step down to smaller models on the
            # GPU before giving up on it.
            compute_type = _best_cuda_compute_type(ctranslate2)
            names = CUDA_AUTO_MODELS if auto else (s.whisper_model,)
            attempts += [("cuda", compute_type, name) for name in names]
        # On CPU, "small" takes 1.3-2 s per pass (captions 3.5-6 s behind);
        # "base" keeps within ~3 s at some cost in accuracy.
        attempts.append(("cpu", "int8", "base" if auto else s.whisper_model))
        for device, compute_type, name in attempts:
            try:
                self.log(
                    f"Loading Whisper '{name}' on {device.upper()} ({compute_type}) (the first run downloads it)..."
                )
                model = WhisperModel(
                    name, device=device, compute_type=compute_type,
                    # Leave two cores for audio I/O, the voice and the video output.
                    cpu_threads=min(8, max(2, (os.cpu_count() or 4) - 2)), download_root=str(WHISPER_DIR),
                )
                # Warm-up: surfaces missing CUDA/cuDNN libraries now rather
                # than on the first spoken word.
                segments, _ = model.transcribe(
                    np.zeros(SR, dtype=np.float32), language="en", beam_size=1, without_timestamps=True
                )
                list(segments)
                return model, f"{device} ({name}, {compute_type})"
            except Exception as e:
                if device == "cpu":
                    raise
                self.log(f"Whisper '{name}' failed on the GPU ({e}) — trying the next option.")
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

    def _language(self, audio, final):
        """Source language for this pass, or None to skip a partial.
        Detection costs a whole extra encoder pass and is unreliable on
        under ~2 s of audio, so it runs once per utterance, as soon as
        there is LANG_DETECT_S of it; until then partials reuse the previous
        utterance's language (and are skipped if there is none yet)."""
        s = self.settings
        if s.source_lang != "auto":
            return s.source_lang
        if self._utt_lang:
            return self._utt_lang
        if len(audio) >= LANG_DETECT_S * SR or (final and not self._last_lang):
            _lang, _prob, probs = self.model.detect_language(audio)
            allowed = [(lang, p) for lang, p in probs if lang in LANGS]
            self._utt_lang = max(allowed, key=lambda lp: lp[1])[0] if allowed else "en"
            self._last_lang = self._utt_lang
            return self._utt_lang
        return self._last_lang

    def _transcribe(self, audio, final):
        s = self.settings
        lang = self._language(audio, final)
        if lang is None:
            return None
        started = time.monotonic()
        prompt_lang, prompt_text = self._prompt
        segments, _info = self.model.transcribe(
            audio,
            language=lang,
            task="transcribe",
            beam_size=self._final_beam if final else 1,
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
        if not final:
            # Shown in the status bar so the operator can see whether the
            # machine keeps up (partials refresh about once per pass).
            took = time.monotonic() - started
            self._pass_s = took if self._pass_s is None else 0.7 * self._pass_s + 0.3 * took
            self.on_status(f"Listening · {self._pass_s:.1f} s/pass")
        if any(marker in text.lower() for marker in HALLUCINATION_MARKERS):
            text = ""
        self._last_source = (lang, text)
        if final:
            self._remember_prompt(lang, text)
        if text and lang in self.translators:
            text = self.translators[lang].translate(text)
        return text

    def _remember_prompt(self, lang, text):
        """The last final (source-language) text primes the next utterance."""
        if text:
            self._prompt = (lang, text[-PROMPT_CHARS:])


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
        # Segmentation: by default Azure ends a sentence after ~0.5 s of
        # silence — mid-sentence at every hesitation, while fast continuous
        # conversation (no such pauses) became one 30 s block. "Semantic" ends
        # sentences by meaning instead (it can't be combined with a silence
        # timeout); the short fragments it still leaves at long hesitations
        # are joined below (see merge_short).
        config.set_property(speechsdk.PropertyId.Speech_SegmentationStrategy, AZURE_SEGMENT_STRATEGY)
        targets = [s.target_lang] + [lang for lang in s.extra_target_langs if lang != s.target_lang]
        for lang in targets:
            config.add_target_language(AZURE_TARGETS[lang])

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

        def source_lang(result):
            detected = result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
            ) or AZURE_LOCALES.get(s.source_lang, "")
            return detected[:2].lower()

        def text_in(result, lang):
            if source_lang(result) == lang:
                return result.text  # already in that language
            wanted = AZURE_TARGETS[lang].lower()
            translations = result.translations or {}
            return next((t for code, t in translations.items() if code.lower() == wanted), "")

        held = {"info": None, "since": 0.0, "activity": 0.0}  # a short final waiting to be joined
        held_lock = threading.Lock()

        def on_recognizing(evt):
            held["activity"] = time.monotonic()  # the person is still talking
            self.on_caption(text_in(evt.result, s.target_lang), False)

        def emit(info):
            self.on_caption(info["texts"][s.target_lang], True, info)

        def join(a, b):
            return {
                "lang": a["lang"],
                "source": f"{a['source']} {b['source']}",
                "texts": {lang: f"{a['texts'][lang]} {b['texts'][lang]}".strip() for lang in targets},
            }

        def merge_short(info):
            with held_lock:
                pending, held["info"] = held["info"], None
            if pending is not None:
                if pending["lang"] == info["lang"]:
                    emit(join(pending, info))
                    return
                emit(pending)
            if len(info["source"].split()) <= AZURE_SHORT_WORDS:
                with held_lock:
                    now = time.monotonic()
                    held["info"], held["since"], held["activity"] = info, now, now
            else:
                emit(info)

        def flush_held(force=False):
            with held_lock:
                if held["info"] is None:
                    return
                now = time.monotonic()
                idle = now - held["activity"] >= AZURE_MERGE_IDLE_S
                if not (force or idle or now - held["since"] >= AZURE_MERGE_MAX_S):
                    return
                pending, held["info"] = held["info"], None
            emit(pending)

        def letters(text):
            return sum(ch.isalpha() for ch in text)

        def on_recognized(evt):
            if evt.result.reason != speechsdk.ResultReason.TranslatedSpeech:
                return
            r = evt.result
            info = {"lang": source_lang(r), "source": r.text, "texts": {lang: text_in(r, lang) for lang in targets}}
            # A final under two letters ("E.", "Hm.") — in the source or the
            # captions — is noise or a cut-off word, not a caption.
            if letters(info["source"]) >= 2 and letters(info["texts"][s.target_lang]) >= 2:
                merge_short(info)

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
        self.log(f"Listening (Azure Speech, region {region}, {s.source_lang} -> {', '.join(targets)}).")
        self.on_status("Listening.")
        try:
            while not self.stop_event.is_set():
                block = self.capture.read(timeout=0.1)
                if block is not None and len(block):
                    stream.write((np.clip(block, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
                flush_held()
        finally:
            stream.close()
            recognizer.stop_continuous_recognition()
            flush_held(force=True)
