"""Live caption layout and rendering for the audio (speech-to-text) mode.

CaptionBoard holds what the speech engine has said so far (finalized text +
the in-progress partial) and lays it out as subtitle lines, in one of three
modes (CaptionConfig.mode): pop-on pages of finished sentences, a box that
fills word by word with confirmed text, or a live roll-up.
render_keyed_captions() draws those lines as white-on-black, exactly like the
caption signal the NDI mode receives, and runs them through the regular keyer
(process_frame), so shadow/color/box follow the same TEXT/BACKGROUND
settings."""

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compositor import KeyerConfig, box_rect, process_frame

LINE_SPACING = 1.25
MAX_COMMITTED = 6
# Pop-on paging: a page stays up at least its reading time (subtitle
# convention ~17 characters/s, never under PAGE_MIN_S); when more text is
# already waiting, pages move CATCH_UP_FACTOR x faster (never under
# PAGE_MIN_FAST_S).
READING_CPS = 17.0
PAGE_MIN_S = 1.5
PAGE_MIN_FAST_S = 1.0
CATCH_UP_FACTOR = 1.5
# Box mode: confirmed text is written in word by word — at least
# FLOW_WORDS_PER_S, faster when more is waiting (all of it within
# FLOW_CATCH_UP_S). A trailing word without a space or punctuation after it
# may still be growing ("deslig" + "uem"): it waits up to FLOW_WORD_WAIT_S.
FLOW_WORDS_PER_S = 4.0
FLOW_CATCH_UP_S = 1.5
FLOW_WORD_WAIT_S = 0.8
FLOW_KEEP_WORDS = 400
SENTENCE_PUNCT = ".,!?;:…"

MODE_SENTENCES = "sentences"
MODE_BOX = "box"
MODE_LIVE = "live"

if sys.platform == "win32":
    _FONT_DIR = Path("C:/Windows/Fonts")
    _FONTS = {False: ("segoeui.ttf", "arial.ttf"), True: ("segoeuib.ttf", "arialbd.ttf")}
else:
    _FONT_DIR = Path("/System/Library/Fonts/Supplemental")
    _FONTS = {False: ("Arial.ttf", "Helvetica.ttc"), True: ("Arial Bold.ttf", "Helvetica.ttc")}
_FALLBACK_DIRS = (Path("/Library/Fonts"), Path("/System/Library/Fonts"), Path("/usr/share/fonts/truetype/dejavu"))


@dataclass
class CaptionConfig:
    font_size: float = 54.0
    bold: bool = False
    max_lines: float = 2.0
    # Maximum text line width, as a percentage of the output canvas width.
    line_width_pct: float = 80.0
    # Distance (px) from the bottom of the canvas to the baseline area of the
    # last line.
    bottom_margin: float = 80.0
    # Captions are cleared after this many seconds without new speech.
    hold_s: float = 4.0
    # MODE_SENTENCES: only finished sentences, never changing once up.
    # MODE_BOX: confirmed text written in as it comes (never corrected); the
    # box clears when full and goes on. MODE_LIVE: text appears while the
    # person speaks and gets corrected as recognition improves.
    mode: str = MODE_SENTENCES


_font_cache = {}


def load_font(size, bold):
    key = (int(size), bool(bold))
    if key in _font_cache:
        return _font_cache[key]
    font = None
    for name in _FONTS[bool(bold)]:
        for folder in (_FONT_DIR, *_FALLBACK_DIRS):
            path = folder / name
            if path.is_file():
                try:
                    font = ImageFont.truetype(str(path), key[0])
                    break
                except OSError:
                    continue
        if font:
            break
    if font is None:
        font = ImageFont.load_default(size=key[0])
    _font_cache[key] = font
    return font


def wrap_lines(text, font, max_width):
    """Greedy word wrap by rendered pixel width."""
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if current and font.getlength(candidate) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


class CaptionBoard:
    """Thread-safe store of the captions: the speech engine calls update()
    from its thread, the output loop calls visible_lines() from another.

    Three layouts:
    - roll-up (live mode): all recent text, the last `max_lines` lines of it,
      updating as the partial changes;
    - pop-on (finished sentences only): each sentence on its own, never
      mixed with the previous one; a sentence longer than `max_lines` is
      split into pages shown one after another, each up for a reading time
      (shorter when more text is waiting, so it doesn't fall behind);
    - box: text the engine has confirmed (stream(), or each final when the
      engine doesn't stream) is written in word by word, filling the lines
      from the top; when full it clears and goes on from the next word.
      Nothing shown ever changes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._committed = []
        self._partial = ""
        self._last_change = 0.0
        # pop-on: finals not yet paged (bounded: unused while in live mode)
        self._sentences = deque(maxlen=MAX_COMMITTED)
        self._pages = deque()  # pop-on: pages (line tuples) waiting
        self._page = None  # pop-on: (lines, shown_at, min_duration)
        self._reset_box()

    def _reset_box(self):
        self._box_text = ""  # all confirmed text (trimmed to FLOW_KEEP_WORDS)
        self._box_streamed = False  # the engine streams: ignore its finals
        self._box_input = 0.0  # when text last came in
        self._box_shown = 0  # words written in so far
        self._box_page = 0  # word the current page starts at
        self._box_credit = 0.0  # words owed to the reveal
        self._box_idle = True  # nothing was waiting to be written in
        self._box_tick = 0.0
        self._box_revealed_at = 0.0

    def update(self, text, final):
        text = text.strip()
        with self._lock:
            if final:
                if text:
                    self._committed.append(text)
                    del self._committed[:-MAX_COMMITTED]
                    self._sentences.append(text)
                    if not self._box_streamed:
                        self._box_append(" " + text)
                self._partial = ""
            else:
                self._partial = text
            self._last_change = time.monotonic()

    def stream(self, text):
        """Confirmed text as the engine streams it (pieces, spacing
        included: " Boa", " no", "ite,"), for the box mode."""
        with self._lock:
            self._box_streamed = True
            self._box_append(text)

    def _box_append(self, text):
        self._box_text += text
        self._box_input = time.monotonic()
        words = self._box_text.split()
        drop = min(self._box_page, len(words) - FLOW_KEEP_WORDS)
        if drop > 0:  # forget pages long gone
            tail = " " if self._box_text[-1:].isspace() else ""
            self._box_text = " ".join(words[drop:]) + tail
            self._box_page -= drop
            self._box_shown -= drop

    def clear(self):
        with self._lock:
            self._committed.clear()
            self._partial = ""
            self._sentences.clear()
            self._pages.clear()
            self._page = None
            self._reset_box()

    def visible_lines(self, font, max_width, max_lines, hold_s, mode=MODE_SENTENCES):
        if mode == MODE_SENTENCES:
            return self._pop_on_lines(font, max_width, max(1, int(max_lines)), hold_s)
        if mode == MODE_BOX:
            return self._box_lines(font, max_width, max(1, int(max_lines)), hold_s)
        with self._lock:
            if time.monotonic() - self._last_change > hold_s:
                self._committed.clear()
                self._partial = ""
            text = " ".join(self._committed + ([self._partial] if self._partial else []))
        if not text:
            return ()
        return tuple(wrap_lines(text, font, max_width)[-max(1, int(max_lines)):])

    def _pop_on_lines(self, font, max_width, max_lines, hold_s):
        with self._lock:
            now = time.monotonic()
            page = self._page
            if page is None or now - page[1] >= page[2]:
                if not self._pages and self._sentences:
                    lines = wrap_lines(self._sentences.popleft(), font, max_width)
                    self._pages.extend(
                        tuple(lines[i:i + max_lines]) for i in range(0, len(lines), max_lines)
                    )
                if self._pages:
                    lines = self._pages.popleft()
                    duration = max(PAGE_MIN_S, sum(len(line) for line in lines) / READING_CPS)
                    if self._pages or self._sentences:
                        duration = max(PAGE_MIN_FAST_S, duration / CATCH_UP_FACTOR)
                    self._page = (lines, now, duration)
                elif page is not None and now - page[1] >= page[2] + hold_s:
                    self._page = None
            return self._page[0] if self._page else ()

    def _box_lines(self, font, max_width, max_lines, hold_s):
        with self._lock:
            now = time.monotonic()
            text = self._box_text
            words = text.split()
            ready = len(words)
            if words and not text[-1].isspace() and text[-1] not in SENTENCE_PUNCT \
                    and now - self._box_input < FLOW_WORD_WAIT_S:
                ready -= 1  # the last word may still grow
            backlog = ready - self._box_shown
            if backlog > 0:
                if self._box_idle:
                    self._box_credit = 1.0  # the first word goes up at once
                    self._box_idle = False
                else:
                    rate = max(FLOW_WORDS_PER_S, backlog / FLOW_CATCH_UP_S)
                    self._box_credit += (now - self._box_tick) * rate
                step = min(backlog, int(self._box_credit))
                if step:
                    self._box_shown += step
                    self._box_credit -= step
                    self._box_revealed_at = now
            else:
                self._box_credit = 0.0
                self._box_idle = True
            self._box_tick = now
            if self._box_shown >= ready and now - max(self._box_input, self._box_revealed_at) > hold_s:
                self._box_page = self._box_shown  # silence: clear; go on on a fresh page
                return ()
            while True:
                lines = wrap_lines(" ".join(words[self._box_page:self._box_shown]), font, max_width)
                if len(lines) <= max_lines:
                    break
                # Full: a new page starts with the words that didn't fit.
                self._box_page += sum(len(line.split()) for line in lines[:max_lines])
            if not lines:
                return ()
            # Fill from the top line down (the renderer stacks lines up from
            # the bottom margin, so pad with empty lines below).
            return tuple(lines) + ("",) * (max_lines - len(lines))


def _render_gray(lines, canvas_w, canvas_h, cap_cfg, font):
    """White text on black, centered horizontally and stacked up from the
    bottom margin, as a PIL "L" image."""
    img = Image.new("L", (canvas_w, canvas_h), 0)
    if lines:
        draw = ImageDraw.Draw(img)
        line_h = cap_cfg.font_size * LINE_SPACING
        bottom = canvas_h - cap_cfg.bottom_margin
        top = bottom - line_h * len(lines)
        for i, line in enumerate(lines):
            y = top + i * line_h + line_h / 2.0
            draw.text((canvas_w / 2.0, y), line, fill=255, font=font, anchor="mm")
    return img


def _union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def render_keyed_captions(lines, canvas_w, canvas_h, cap_cfg, font, keyer_cfg: KeyerConfig):
    """Renders the caption lines and keys them like the NDI caption signal
    (shadow, text color, background box), returning canvas_w x canvas_h
    BGRA. Only the region holding the text (+ shadow reach) and the box is
    run through the keyer — a few times faster than the full frame."""
    cfg = keyer_config_for_captions(keyer_cfg)
    img = _render_gray(lines, canvas_w, canvas_h, cap_cfg, font)
    region = None
    text_bbox = img.getbbox()
    if text_bbox:
        reach = abs(cfg.shadow_dx) + abs(cfg.shadow_dy) + 3 * cfg.shadow_blur_sigma + 2
        x0, y0, x1, y1 = text_bbox
        region = (x0 - reach, y0 - reach, x1 + reach, y1 + reach)
    if cfg.box_enabled and cfg.box_opacity > 0:
        region = _union(region, box_rect((canvas_h, canvas_w), cfg))

    out = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    if region is None:
        return out
    x0, y0 = max(0, int(region[0])), max(0, int(region[1]))
    x1, y1 = min(canvas_w, int(np.ceil(region[2]))), min(canvas_h, int(np.ceil(region[3])))
    if x1 <= x0 or y1 <= y0:
        return out

    gray = np.asarray(img)[y0:y1, x0:x1]
    crop = np.empty((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    crop[:, :, 0] = gray
    crop[:, :, 1] = gray
    crop[:, :, 2] = gray
    crop[:, :, 3] = 255
    out[y0:y1, x0:x1] = process_frame(crop, cfg, full_shape=(canvas_h, canvas_w), origin=(x0, y0))
    return out


def keyer_config_for_captions(cfg: KeyerConfig) -> KeyerConfig:
    """The rendered captions are clean white-on-black, so the keyer always
    uses a full-range luma key on them regardless of the NDI key settings
    (shadow, colors and box are kept)."""
    return replace(cfg, key_mode="luma", black_level=0.0, white_level=255.0)
