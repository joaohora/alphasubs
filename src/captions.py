"""Live caption layout and rendering for the audio (speech-to-text) mode.

CaptionBoard holds what the speech engine has said so far (finalized text +
the in-progress partial) and lays it out as roll-up subtitle lines.
render_keyed_captions() draws those lines as white-on-black, exactly like the
caption signal the NDI mode receives, and runs them through the regular keyer
(process_frame), so shadow/color/box follow the same TEXT/BACKGROUND
settings."""

import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compositor import KeyerConfig, box_rect, process_frame

LINE_SPACING = 1.25
MAX_COMMITTED = 6

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
    # False: only finished sentences are shown, and never change once up.
    # True: text also appears while the person speaks, and gets corrected
    # as recognition improves (faster, but visibly rewrites itself).
    live: bool = False


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
    from its thread, the output loop calls visible_lines() from another."""

    def __init__(self):
        self._lock = threading.Lock()
        self._committed = []
        self._partial = ""
        self._last_change = 0.0

    def update(self, text, final):
        text = text.strip()
        with self._lock:
            if final:
                if text:
                    self._committed.append(text)
                    del self._committed[:-MAX_COMMITTED]
                self._partial = ""
            else:
                self._partial = text
            self._last_change = time.monotonic()

    def clear(self):
        with self._lock:
            self._committed.clear()
            self._partial = ""

    def visible_lines(self, font, max_width, max_lines, hold_s):
        with self._lock:
            if time.monotonic() - self._last_change > hold_s:
                self._committed.clear()
                self._partial = ""
            text = " ".join(self._committed + ([self._partial] if self._partial else []))
        if not text:
            return ()
        return tuple(wrap_lines(text, font, max_width)[-max(1, int(max_lines)):])


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
