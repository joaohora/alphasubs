"""Offline text translation between EN/ES/PT using Argos Translate's
CTranslate2 models, loaded directly with ctranslate2 (no
argostranslate/stanza/torch dependency). Models (~70-100 MB each) are
downloaded once, on first use, into MODELS_DIR.

Argos packages come with one of two tokenizers: a sentencepiece model
(sentencepiece.model) or, for models derived from OPUS-MT, Moses
tokenization + subword-nmt BPE codes (bpe.model). Both are supported."""

import json
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import ctranslate2

INDEX_URL = "https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json"
MODELS_DIR = Path.home() / ".alphasubs_models" / "translate"
TOKENIZER_FILES = ("sentencepiece.model", "bpe.model")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
# argos-net.com answers 403 to urllib's default User-Agent.
_HEADERS = {"User-Agent": "AlphaSubs/1.0"}


def _open_url(url, timeout):
    return urllib.request.urlopen(urllib.request.Request(url, headers=_HEADERS), timeout=timeout)


def _model_dir(src, tgt):
    return MODELS_DIR / f"{src}_{tgt}"


def _find_package_root(path):
    """The extracted .argosmodel holds one top-level folder with model/ (the
    CTranslate2 model) and its tokenizer file."""
    for name in TOKENIZER_FILES:
        for tok in path.rglob(name):
            if (tok.parent / "model" / "model.bin").is_file():
                return tok.parent
    return None


def ensure_model(src, tgt, log=print):
    """Returns the local folder of the src->tgt model, downloading it first
    if needed."""
    root = _find_package_root(_model_dir(src, tgt)) if _model_dir(src, tgt).is_dir() else None
    if root:
        return root

    with _open_url(INDEX_URL, timeout=30) as resp:
        index = json.load(resp)
    entry = next((p for p in index if p["from_code"] == src and p["to_code"] == tgt), None)
    if entry is None:
        raise RuntimeError(f"No offline translation model available for {src} -> {tgt}.")

    url = entry["links"][0]
    log(f"Downloading translation model {src} -> {tgt} (one-time, ~100 MB)...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=MODELS_DIR) as tmp:
        archive = Path(tmp) / "model.zip"
        with _open_url(url, timeout=60) as resp, open(archive, "wb") as f:
            shutil.copyfileobj(resp, f)
        staging = Path(tmp) / "x"
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staging)
        target = _model_dir(src, tgt)
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(staging), str(target))
    root = _find_package_root(_model_dir(src, tgt))
    if root is None:
        raise RuntimeError(f"Downloaded translation model {src} -> {tgt} looks incomplete.")
    log(f"Translation model {src} -> {tgt} ready.")
    return root


class _SentencePieceTokenizer:
    def __init__(self, model_file):
        import sentencepiece

        self._sp = sentencepiece.SentencePieceProcessor(model_file=str(model_file))

    def encode(self, sentence):
        return self._sp.encode(sentence, out_type=str)

    def decode(self, tokens):
        # Target pieces aren't all in the source sentencepiece model, so
        # detokenize by hand (as Argos does) instead of with sp.decode().
        return "".join(tokens).replace("▁", " ").strip()


class _BPETokenizer:
    def __init__(self, codes_file, src, tgt):
        from sacremoses import MosesDetokenizer, MosesTokenizer
        from subword_nmt.apply_bpe import BPE

        self._tokenizer = MosesTokenizer(lang=src)
        self._detokenizer = MosesDetokenizer(lang=tgt)
        with open(codes_file, encoding="utf-8") as f:
            self._bpe = BPE(f)

    def encode(self, sentence):
        words = self._tokenizer.tokenize(sentence, aggressive_dash_splits=True, escape=True)
        return self._bpe.segment_tokens(words)

    def decode(self, tokens):
        words = " ".join(tokens).replace("@@ ", "").split()
        return self._detokenizer.detokenize(words).replace(" @-@ ", "-").strip()


class Translator:
    def __init__(self, src, tgt, log=print):
        root = ensure_model(src, tgt, log=log)
        # Runs on CPU on purpose: these models are small and fast, and the
        # GPU is better left to Whisper.
        self._translator = ctranslate2.Translator(str(root / "model"), device="cpu", compute_type="int8")
        if (root / "sentencepiece.model").is_file():
            self._tok = _SentencePieceTokenizer(root / "sentencepiece.model")
        else:
            self._tok = _BPETokenizer(root / "bpe.model", src, tgt)

    def translate(self, text):
        sentences = [s for s in _SENTENCE_SPLIT.split(text.strip()) if s]
        if not sentences:
            return ""
        tokens = [self._tok.encode(s) for s in sentences]
        results = self._translator.translate_batch(tokens, beam_size=2, max_decoding_length=256)
        out = [self._tok.decode(r.hypotheses[0]) for r in results]
        return " ".join(s for s in out if s)
