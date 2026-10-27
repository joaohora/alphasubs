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
- SonioxEngine / SpeechmaticsEngine: cloud streaming recognition (+ their
  own translation) over a WebSocket; need an API key.

Any engine's finals can instead be translated by Claude (ClaudeTranslation):
slower by a request per sentence, but it translates with the previous
sentences as context, into European Portuguese.

Heavy imports (faster_whisper, azure SDK) happen inside the engine thread so
the app starts fast and NDI-only use doesn't need them."""

import json
import os
import queue
import re
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import audio_io
import languages

SR = audio_io.SAMPLE_RATE
LANGS = languages.BASIC  # what the local, Azure and Speechmatics engines handle
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
SONIOX_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
SONIOX_MODEL = "stt-rt-v5"
# Silence after which Soniox ends an utterance (500-3000 ms).
SONIOX_ENDPOINT_DELAY_MS = 500
SPEECHMATICS_URL = "wss://eu.rt.speechmatics.com/v2/"
SPEECHMATICS_MAX_DELAY_S = 2.0
SPEECHMATICS_END_SILENCE_S = 0.8
# The cloud services send an utterance's translation after its source text:
# a finished utterance waits at most this long for it.
TRANSLATION_WAIT_S = 2.0
SENTENCE_END = (".", "?", "!", "…")
# An utterance of at most FRAGMENT_WORDS waits to be joined to the next one
# while the speaker keeps talking (see _CloudStreamEngine._emit).
FRAGMENT_WORDS = 5
HOLD_IDLE_S = 1.2
HOLD_MAX_S = 15.0
RECONNECT_DELAY_S = 2.0
CLAUDE_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")
# Server-side fallback when a model declines a request (not for Haiku).
CLAUDE_FALLBACK_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5")
CLAUDE_CONTEXT_SENTENCES = 6
CLAUDE_TIMEOUT_S = 10.0
CLAUDE_SLOW_S = 3.0
LANG_NAMES = {**languages.NAMES, "pt": "European Portuguese (Portugal)"}
CLAUDE_SYSTEM = (
    "You are the translator of live subtitles at a public event. Each request has one new sentence "
    "from a speech-recognition system, transcribed as the speaker talks, and the previous sentences "
    "with their translations as context. Translate only the new sentence into {target}.\n"
    "- Write natural, fluent subtitles that keep the speaker's meaning and tone, not a word-for-word "
    "translation.\n"
    "- Use the context to resolve pronouns, terms and sentences that the recognition split in two.\n"
    "- The recognition makes mistakes: when a word clearly doesn't fit, translate what the speaker "
    "obviously meant. Don't add anything that wasn't said.\n"
    "- Keep the listed names and terms exactly as written.\n"
    "{variant}"
    "- Reply with the translation only: no quotes, notes or explanations. If the new sentence is "
    "only a filler or noise (\"uh\", \"hmm\"), reply with nothing."
)
CLAUDE_PT_VARIANT = (
    "- Portuguese means the European Portuguese of Portugal, never Brazilian: \"está a fazer\", "
    "\"vocês\"/\"os senhores\", \"ecrã\", \"equipa\", \"autocarro\", European spelling and vocabulary.\n"
)
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
    engine: str = "local"  # "local" | "azure" | "soniox" | "speechmatics"
    device_index: int = None
    # An NDI source to take the audio from instead of a sound device.
    ndi_source: object = None
    channel: int = None  # 0-based, or None = mix of the first two channels
    source_lang: str = "auto"  # "auto" | a language code (see languages.py)
    target_lang: str = "pt"  # "pt" | "en"
    whisper_model: str = "auto"
    # Comma-separated names/terms to bias recognition towards.
    hotwords: str = ""
    azure_key: str = ""
    azure_region: str = ""
    soniox_key: str = ""
    speechmatics_key: str = ""
    # Who translates: "engine" (the recognition engine itself) | "claude".
    translation: str = "engine"
    claude_key: str = ""
    claude_model: str = CLAUDE_MODELS[0]
    # Further languages to translate finals into, besides target_lang (the
    # voice's language). Only Azure and Claude use it — they translate to
    # several languages at once; local consumers translate `info["source"]`
    # instead.
    extra_target_langs: tuple = ()


def _best_cuda_compute_type(ctranslate2):
    try:
        supported = ctranslate2.get_supported_compute_types("cuda")
    except Exception:
        return "float16"
    return next((t for t in CUDA_COMPUTE_TYPES if t in supported), "float32")


def create_engine(settings, on_caption, log, on_status):
    engines = {
        "local": LocalWhisperEngine, "azure": AzureEngine,
        "soniox": SonioxEngine, "speechmatics": SpeechmaticsEngine,
    }
    return engines.get(settings.engine, LocalWhisperEngine)(settings, on_caption, log, on_status)


def _tidy(text):
    """Joins streamed pieces' spacing: single spaces, none before punctuation."""
    return re.sub(r"\s+([.,!?;:%])", r"\1", " ".join(text.split()))


class ClaudeTranslation:
    """Translates an engine's finals with Claude, in order, each with the
    previous sentences (and their translations) as context. Stands in for
    the engine's on_caption: partials are dropped (they would show the
    source language until the translation arrives), finals go out once
    translated. Sentences that pile up while a request runs are translated
    together, so a slow moment doesn't build a growing delay. On an error
    the engine's own text is used (its translation, or the source)."""

    def __init__(self, settings, on_caption, log):
        self.settings = settings
        self.on_caption = on_caption
        self.log = log
        self._queue = queue.Queue()
        self._history = {}  # target language -> deque of (source, translation)
        self._client = None
        self._last_error = None
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        import anthropic

        key = self.settings.claude_key.strip()
        if not key:
            raise RuntimeError("Claude translation needs an API key (Input panel).")
        self._client = anthropic.Anthropic(api_key=key, timeout=CLAUDE_TIMEOUT_S, max_retries=1)
        self._thread.start()

    def close(self):
        self._queue.put(None)

    def __call__(self, text, final, info=None):
        if not final:
            return
        if not info or not info.get("source"):
            self.on_caption(text, final, info)
            return
        self._queue.put(info)

    def _loop(self):
        while True:
            batch = [self._queue.get()]
            while True:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            closing = None in batch
            batch = [info for info in batch if info is not None]
            # Consecutive sentences in the same language go out as one.
            groups = []
            for info in batch:
                if groups and groups[-1][-1]["lang"] == info["lang"]:
                    groups[-1].append(info)
                else:
                    groups.append([info])
            for group in groups:
                self._translate_group(group)
            if closing:
                return

    def _translate_group(self, group):
        s = self.settings
        lang = group[0]["lang"]
        source = " ".join(info["source"] for info in group)
        targets = [s.target_lang] + [t for t in s.extra_target_langs if t != s.target_lang]
        texts = {}
        for target in targets:
            if target == lang:
                texts[target] = source
                continue
            fallback = " ".join(info.get("texts", {}).get(target) or info["source"] for info in group)
            texts[target] = self._translate(source, lang, target, fallback)
        if not texts[s.target_lang]:
            return  # only a filler ("uh", "hmm"): nothing to show or say
        info = {"lang": lang, "source": source, "texts": texts}
        self.on_caption(texts[s.target_lang], True, info)

    def _translate(self, source, lang, target, fallback):
        import anthropic

        s = self.settings
        history = self._history.setdefault(target, deque(maxlen=CLAUDE_CONTEXT_SENTENCES))
        parts = []
        terms = s.hotwords.strip()
        if terms:
            parts.append(f"Names and terms (keep exactly as written): {terms}")
        if history:
            lines = "\n".join(f"- {src}\n  → {out}" for src, out in history)
            parts.append(f"Previous sentences and their translations (context only):\n{lines}")
        parts.append(f"New sentence ({LANG_NAMES.get(lang, lang)}):\n{source}")
        system = CLAUDE_SYSTEM.format(
            target=LANG_NAMES.get(target, target), variant=CLAUDE_PT_VARIANT if target == "pt" else ""
        )
        request = dict(
            model=s.claude_model,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": "\n\n".join(parts)}],
            output_config={"effort": "low"},
        )
        started = time.monotonic()
        try:
            if s.claude_model in CLAUDE_FALLBACK_MODELS:
                response = self._client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request
                )
            else:
                response = self._client.messages.create(**request)
        except anthropic.AuthenticationError:
            return self._failed("Claude rejected the API key — check it in the Input panel.", fallback)
        except anthropic.APIStatusError as e:
            return self._failed(f"Claude translation error {e.status_code}: {e.message}", fallback)
        except anthropic.APIConnectionError:
            return self._failed("Claude translation: no connection (timeout or network).", fallback)
        took = time.monotonic() - started
        if took > CLAUDE_SLOW_S:
            self.log(f"Claude translation took {took:.1f} s.")
        if response.stop_reason == "refusal":
            return self._failed("Claude declined to translate a sentence.", fallback)
        text = _tidy("".join(b.text for b in response.content if b.type == "text"))
        self._last_error = None
        history.append((source, text))
        return text

    def _failed(self, message, fallback):
        # Logged once per kind of error, not on every sentence.
        if message != self._last_error:
            self.log(f"{message} Using the engine's own text meanwhile.")
            self._last_error = message
        return fallback


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
        # With Claude translating, the engine only needs to transcribe (it
        # may still translate, as a fallback) and its finals go through
        # Claude on the way out.
        self.translation = None
        if settings.translation == "claude":
            self.translation = ClaudeTranslation(settings, on_caption, log)
            on_caption = self.translation
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
            if self.translation is not None:
                self.translation.start()
                self.log(f"Translating with Claude ({self.settings.claude_model}).")
            self._run()
        except Exception as e:
            self.error = str(e)
            self.log(f"Speech engine error: {e}")
            self.on_status("Speech engine error — see Log.")
        finally:
            if self.capture is not None:
                self.capture.close()
            if self.translation is not None:
                self.translation.close()

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
        for src in sources if self.translation is None else ():
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
        # With Claude translating, the subtitle language may be one Azure
        # isn't set up for: Azure then only provides a fallback.
        targets = [
            lang for lang in [s.target_lang] + list(s.extra_target_langs) if lang in AZURE_TARGETS
        ] or ["en"]
        targets = list(dict.fromkeys(targets))
        caption_langs = list(dict.fromkeys([s.target_lang] + list(s.extra_target_langs)))
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
            if lang not in targets:
                return ""  # Claude translates into it
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
                "texts": {lang: f"{a['texts'][lang]} {b['texts'][lang]}".strip() for lang in caption_langs},
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
            info = {"lang": source_lang(r), "source": r.text, "texts": {lang: text_in(r, lang) for lang in caption_langs}}
            # A final under two letters ("E.", "Hm.") — in the source or the
            # captions — is noise or a cut-off word, not a caption.
            translated = letters(info["texts"][s.target_lang]) >= 2 or self.translation is not None
            if letters(info["source"]) >= 2 and translated:
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


def _join_translations(first, second):
    """Joins a held fragment's translation to the next utterance's — unless
    the service already retranslated the fragment as the start of the next
    one ("Antes." + "Antes de começarmos, ...")."""
    def words(pieces):
        return re.sub(r"[^\w\s]", "", "".join(pieces)).lower().split()

    head = words(first)
    if head and words(second)[:len(head)] == head:
        return list(second)
    return list(first) + [" "] + list(second)


class _Utterances:
    """Assembles the utterances of a cloud streaming service, which sends an
    utterance's source text and, separately and a little later, its
    translation. A finished utterance waits (up to TRANSLATION_WAIT_S) until
    its translation is in."""

    def __init__(self):
        self._source = []  # final source pieces of the utterance being spoken
        self._langs = []
        self._needs_translation = False
        self._translation = []  # final translation pieces that came meanwhile
        self._partial = ""  # translation still changing (not final)
        self._waiting = None  # a finished utterance waiting for its translation

    def add_source(self, text, lang=None, needs_translation=False):
        self._source.append(text)
        if lang:
            self._langs.append(lang)
        self._needs_translation |= needs_translation

    def add_translation(self, text):
        (self._waiting["translation"] if self._waiting else self._translation).append(text)

    def set_partial_translation(self, text):
        self._partial = text

    def partial_text(self, partial_source, translating):
        if translating:
            return _tidy("".join(self._translation) + self._partial)
        return _tidy("".join(self._source) + partial_source)

    def end(self):
        """Ends the current utterance; returns the utterances now complete."""
        done = []
        source = _tidy("".join(self._source))
        if source and self._waiting is not None:
            done.append(self._waiting)  # the next one ended first: no longer wait
            self._waiting = None
        if source:
            lang = Counter(self._langs).most_common(1)[0][0] if self._langs else None
            item = {
                "source": source, "lang": lang, "translation": self._translation,
                "since": time.monotonic(),
            }
            if self._needs_translation:
                self._waiting = item
            else:
                done.append(item)
        self._source, self._langs, self._translation = [], [], []
        self._needs_translation = False
        return done

    def ready(self, force=False):
        """The waiting utterance, once its translation is in (or late)."""
        w = self._waiting
        if w is None:
            return []
        # Complete once the translated sentence has ended (its translation
        # may come in several pieces, and the next sentence's may follow).
        late = time.monotonic() - w["since"] >= TRANSLATION_WAIT_S
        if force or late or "".join(w["translation"]).rstrip().endswith(SENTENCE_END):
            self._waiting = None
            return [w]
        return []


class _CloudStreamEngine(_Engine):
    """A cloud recognizer over a WebSocket: audio goes up as 16-bit PCM,
    results come down as JSON (read on a second thread). If the connection
    drops it reconnects, skipping the audio of the outage; errors that a
    retry can't fix (a wrong key, a bad setting) stop the engine."""

    service = ""

    def _run(self):
        from websockets.exceptions import ConnectionClosed, InvalidStatus

        self._check_settings()
        self.translating = self.translation is None and self.settings.source_lang != self.settings.target_lang
        self._lock = threading.Lock()
        self._utterances = _Utterances()
        self._shown_partial = ""
        self._held = None  # a fragment waiting to be joined (see _emit)
        self._activity = 0.0  # when partial results last came in
        self._fatal = None
        self._open_capture()
        connected_once = False
        while not self.stop_event.is_set():
            try:
                ws = self._connect()
            except InvalidStatus as e:
                status = e.response.status_code
                if status in (401, 403):
                    raise RuntimeError(f"{self.service} rejected the API key (HTTP {status}).")
                self.log(f"{self.service}: connection refused (HTTP {status}) — retrying.")
                self.stop_event.wait(RECONNECT_DELAY_S)
                continue
            except (OSError, TimeoutError) as e:
                self.log(f"{self.service}: can't connect ({e}) — retrying.")
                self.on_status(f"{self.service}: reconnecting...")
                self.stop_event.wait(RECONNECT_DELAY_S)
                continue
            if not connected_once:
                self.log(
                    f"Listening ({self.service}, {self.settings.source_lang} -> {self.settings.target_lang}"
                    f"{'' if self.translation is None else ', translation by Claude'})."
                )
                connected_once = True
            else:
                self.log(f"{self.service}: reconnected.")
                self.capture.read(timeout=0)  # skip the audio of the outage
            self.on_status("Listening.")
            reader = threading.Thread(target=self._read, args=(ws,), daemon=True)
            reader.start()
            try:
                while not self.stop_event.is_set() and reader.is_alive():
                    block = self.capture.read(timeout=0.1)
                    if block is not None and len(block):
                        self._send_audio(ws, (np.clip(block, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
                    with self._lock:
                        self._emit(self._utterances.ready())
                if self.stop_event.is_set():
                    self._finish(ws)
                    reader.join(timeout=3.0)
            except ConnectionClosed:
                pass
            finally:
                ws.close()
                reader.join(timeout=1.0)
                with self._lock:
                    self._emit(self._utterances.end() + self._utterances.ready(force=True), force=True)
            if self._fatal:
                raise RuntimeError(self._fatal)
            if not self.stop_event.is_set():
                self.log(f"{self.service}: connection lost — reconnecting.")
                self.on_status(f"{self.service}: reconnecting...")
                self.stop_event.wait(RECONNECT_DELAY_S)

    def _read(self, ws):
        from websockets.exceptions import ConnectionClosed

        try:
            for raw in ws:
                if isinstance(raw, str):
                    msg = json.loads(raw)
                    with self._lock:
                        self._handle(msg)
        except ConnectionClosed:
            pass
        except Exception as e:
            self.log(f"{self.service}: unexpected message ({e}).")

    def _partial(self, text):
        if text:
            self._activity = time.monotonic()  # the speaker is still talking
        if text and text != self._shown_partial:
            self._shown_partial = text
            self.on_caption(text, False)

    def _emit(self, items, force=False):
        """Outputs finished utterances. A short fragment (the speaker
        hesitated: "No final," + "haverá tempo..."; Soniox even closes one
        as "Before." + "We begin, please...") is held and joined to the next
        utterance. It goes out alone once the speaker has stopped (no
        partial results for HOLD_IDLE_S), or after HOLD_MAX_S at most."""
        for item in items:
            held = self._held
            if held is not None:
                self._held = None
                if held["lang"] == item["lang"]:
                    item = {
                        "source": f"{held['source']} {item['source']}", "lang": item["lang"],
                        "translation": _join_translations(held["translation"], item["translation"]),
                        "since": held["since"],
                    }
                else:
                    self._output(held)
            if len(item["source"].split()) <= FRAGMENT_WORDS:
                self._held = item
            else:
                self._output(item)
        held = self._held
        now = time.monotonic()
        if held is not None and (
            force or now - self._activity >= HOLD_IDLE_S or now - held["since"] >= HOLD_MAX_S
        ):
            self._held = None
            self._output(held)

    def _output(self, item):
        s = self.settings
        self._shown_partial = ""
        lang = item["lang"] if item["lang"] in languages.NAMES else (s.source_lang if s.source_lang != "auto" else "en")
        texts = {}
        if lang == s.target_lang:
            texts[s.target_lang] = item["source"]
        elif item["translation"]:
            texts[s.target_lang] = _tidy("".join(item["translation"]))
        elif self.translation is None:
            self.log(f"{self.service}: no translation came for \"{item['source']}\".")
        self.on_caption(texts.get(s.target_lang, ""), True, {"lang": lang, "source": item["source"], "texts": texts})

    # Per service:
    def _check_settings(self):
        raise NotImplementedError

    def _connect(self):
        raise NotImplementedError

    def _handle(self, msg):
        raise NotImplementedError

    def _send_audio(self, ws, pcm):
        ws.send(pcm)

    def _finish(self, ws):
        raise NotImplementedError


class SonioxEngine(_CloudStreamEngine):
    service = "Soniox"

    def _check_settings(self):
        if not self.settings.soniox_key.strip():
            raise RuntimeError("Soniox needs an API key (Input panel).")

    def _connect(self):
        from websockets.sync.client import connect

        s = self.settings
        ws = connect(
            SONIOX_URL, additional_headers={"Authorization": f"Bearer {s.soniox_key.strip()}"},
            open_timeout=10, max_size=None,
        )
        config = {
            "model": SONIOX_MODEL,
            "audio_format": "pcm_s16le",
            "sample_rate": SR,
            "num_channels": 1,
            # Hints only bias recognition: with Auto, other languages are
            # still detected.
            "language_hints": list(LANGS) if s.source_lang == "auto" else [s.source_lang],
            "enable_language_identification": True,
            "enable_endpoint_detection": True,
            "max_endpoint_delay_ms": SONIOX_ENDPOINT_DELAY_MS,
        }
        if s.source_lang != "auto":
            config["language_hints_strict"] = True
        terms = [t.strip() for t in s.hotwords.split(",") if t.strip()]
        if terms:
            config["context"] = {"terms": terms}
        if self.translating:
            config["translation"] = {"type": "one_way", "target_language": s.target_lang}
        ws.send(json.dumps(config))
        return ws

    def _handle(self, msg):
        if msg.get("error_code") is not None:
            code = msg["error_code"]
            text = f"Soniox error {code}: {msg.get('error_message', '')}"
            self.log(text)
            if code in (400, 401, 402, 403):
                self._fatal = text
            return
        u = self._utterances
        partial_source, partial_translation = [], []
        for token in msg.get("tokens", []):
            text = token.get("text", "")
            final = token.get("is_final", False)
            status = token.get("translation_status", "none")
            if text == "<end>":
                if final:
                    self._emit(u.end())
                continue
            if status == "translation":
                if final:
                    u.add_translation(text)
                else:
                    partial_translation.append(text)
            elif final:
                u.add_source(text, token.get("language"), needs_translation=status == "original")
                # A sentence ends a caption: the short pauses between
                # sentences in a fluent speech don't make an endpoint.
                if text.rstrip().endswith(SENTENCE_END):
                    self._emit(u.end())
            else:
                partial_source.append(text)
        u.set_partial_translation("".join(partial_translation))
        if "".join(partial_source).strip():
            self._activity = time.monotonic()
        self._emit(u.ready())
        self._partial(u.partial_text("".join(partial_source), self.translating))

    def _finish(self, ws):
        ws.send("")  # an empty text frame ends the stream


class SpeechmaticsEngine(_CloudStreamEngine):
    service = "Speechmatics"

    def _check_settings(self):
        s = self.settings
        if not s.speechmatics_key.strip():
            raise RuntimeError("Speechmatics needs an API key (Input panel).")
        if s.source_lang == "auto":
            raise RuntimeError(
                "Speechmatics can't detect the spoken language in real time — choose it in the Input panel."
            )

    def _connect(self):
        from websockets.sync.client import connect

        s = self.settings
        ws = connect(
            SPEECHMATICS_URL, additional_headers={"Authorization": f"Bearer {s.speechmatics_key.strip()}"},
            open_timeout=10, max_size=None,
        )
        transcription = {
            "language": s.source_lang,
            "operating_point": "enhanced",
            "max_delay": SPEECHMATICS_MAX_DELAY_S,
            "enable_partials": True,
            "conversation_config": {"end_of_utterance_silence_trigger": SPEECHMATICS_END_SILENCE_S},
        }
        terms = [t.strip() for t in s.hotwords.split(",") if t.strip()]
        if terms:
            transcription["additional_vocab"] = [{"content": t} for t in terms]
        start = {
            "message": "StartRecognition",
            "audio_format": {"type": "raw", "encoding": "pcm_s16le", "sample_rate": SR},
            "transcription_config": transcription,
        }
        if self.translating:
            start["translation_config"] = {"target_languages": [s.target_lang], "enable_partials": True}
        ws.send(json.dumps(start))
        self._seq_no = 0
        return ws

    def _send_audio(self, ws, pcm):
        ws.send(pcm)
        self._seq_no += 1

    def _handle(self, msg):
        kind = msg.get("message")
        u = self._utterances
        if kind == "AddTranscript":
            text = msg.get("metadata", {}).get("transcript", "")
            if text.strip():
                u.add_source(" " + text, self.settings.source_lang, needs_translation=self.translating)
                # A sentence ends a caption: the short pauses between
                # sentences in a fluent speech don't make an EndOfUtterance.
                if text.rstrip().endswith(SENTENCE_END):
                    self._emit(u.end())
        elif kind == "AddPartialTranscript":
            if msg.get("metadata", {}).get("transcript", "").strip():
                self._activity = time.monotonic()
            self._partial(u.partial_text(" " + msg.get("metadata", {}).get("transcript", ""), self.translating))
        elif kind == "AddTranslation":
            text = " ".join(r.get("content", "") for r in msg.get("results", []))
            u.set_partial_translation("")
            if text.strip():
                u.add_translation(" " + text)
        elif kind == "AddPartialTranslation":
            text = " ".join(r.get("content", "") for r in msg.get("results", []))
            u.set_partial_translation(" " + text if text.strip() else "")
            self._partial(u.partial_text("", True))
        elif kind == "EndOfUtterance":
            self._emit(u.end())
        elif kind == "Error":
            text = f"Speechmatics error ({msg.get('type')}): {msg.get('reason', '')}"
            self.log(text)
            if msg.get("type") not in ("internal_error", "job_error", "idle_timeout"):
                self._fatal = text
        elif kind == "Warning":
            self.log(f"Speechmatics warning ({msg.get('type')}): {msg.get('reason', '')}")
        self._emit(u.ready())

    def _finish(self, ws):
        ws.send(json.dumps({"message": "EndOfStream", "last_seq_no": self._seq_no}))
