"""The languages the app works in, per engine.

The local (Whisper + offline translation), Azure and Speechmatics engines
are set up for English, Portuguese and Spanish. Soniox recognizes 60
languages and translates between any two of them; Claude translates
anything too; Soniox voices speak all 60.

Captions are drawn with a standard font (Segoe UI / Arial), left to right,
wrapped at spaces — so only languages written that way, in the Latin,
Cyrillic or Greek alphabet, can be subtitle languages. Spoken (and voice)
languages have no such limit."""

BASIC = ("en", "pt", "es")

# Soniox real-time recognition / translation / voices.
NAMES = {
    "af": "Afrikaans", "sq": "Albanian", "ar": "Arabic", "az": "Azerbaijani", "eu": "Basque",
    "be": "Belarusian", "bn": "Bengali", "bs": "Bosnian", "bg": "Bulgarian", "ca": "Catalan",
    "zh": "Chinese", "hr": "Croatian", "cs": "Czech", "da": "Danish", "nl": "Dutch", "en": "English",
    "et": "Estonian", "fi": "Finnish", "fr": "French", "gl": "Galician", "de": "German", "el": "Greek",
    "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hu": "Hungarian", "id": "Indonesian",
    "it": "Italian", "ja": "Japanese", "kn": "Kannada", "kk": "Kazakh", "ko": "Korean", "lv": "Latvian",
    "lt": "Lithuanian", "mk": "Macedonian", "ms": "Malay", "ml": "Malayalam", "mr": "Marathi",
    "no": "Norwegian", "fa": "Persian", "pl": "Polish", "pt": "Portuguese", "pa": "Punjabi",
    "ro": "Romanian", "ru": "Russian", "sr": "Serbian", "sk": "Slovak", "sl": "Slovenian",
    "es": "Spanish", "sw": "Swahili", "sv": "Swedish", "tl": "Tagalog", "ta": "Tamil", "te": "Telugu",
    "th": "Thai", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese", "cy": "Welsh",
}
# Right-to-left, without spaces between words, or in scripts the caption
# font doesn't have.
_NOT_CAPTIONABLE = {
    "ar", "fa", "ur", "he", "zh", "ja", "ko", "th",
    "bn", "gu", "hi", "kn", "ml", "mr", "pa", "ta", "te",
}
ALL = tuple(NAMES)
CAPTIONABLE = tuple(code for code in NAMES if code not in _NOT_CAPTIONABLE)

# The app's own three keep their usual labels.
_LABELS = {"pt": "Português", "en": "English", "es": "Español"}


def label(code):
    return _LABELS.get(code) or NAMES.get(code, code)


def ordered(codes):
    """Portuguese, English, Spanish first, then the rest by name."""
    first = [c for c in ("pt", "en", "es") if c in codes]
    rest = sorted((c for c in codes if c not in first), key=label)
    return first + rest


def spoken(engine):
    return ALL if engine == "soniox" else BASIC


def subtitles(engine, translation):
    """Subtitle languages: anything captionable when Soniox or Claude
    translates; the other engines translate into Portuguese and English."""
    if engine == "soniox" or translation == "claude":
        return CAPTIONABLE
    return ("pt", "en")


def voices(voice_engine):
    return ALL if voice_engine == "soniox" else ("pt", "en")
