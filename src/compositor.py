"""Background keying (luma or a picked color) + drop shadow, producing
straight-alpha BGRA."""

from dataclasses import dataclass

import cv2
import numpy as np


FPS_PRESETS = {
    "23.98": (24000, 1001),
    "24": (24, 1),
    "25": (25, 1),
    "29.97": (30000, 1001),
    "30": (30, 1),
    "50": (50, 1),
    "59.94": (60000, 1001),
    "60": (60, 1),
}


@dataclass
class OutputConfig:
    # Output canvas resolution (px).
    canvas_w: int = 1920
    canvas_h: int = 1080
    # Output frame rate (numerator/denominator, e.g. 30000/1001 = 29.97fps).
    fps_n: int = 30000
    fps_d: int = 1001
    # Transform applied to the input signal inside the output canvas.
    scale: float = 1.0
    # Offset of the (already scaled) signal in pixels, from the canvas center:
    # 0,0 = centered. pos_x_px positive shifts right. pos_y_px positive shifts
    # UP, negative shifts DOWN (inverted relative to image coordinates — more
    # intuitive for the user).
    pos_x_px: float = 0.0
    pos_y_px: float = 0.0


@dataclass
class KeyerConfig:
    # Which method turns the background into alpha: "luma" (black/white level
    # below) or "color" (distance from key_color_bgr below).
    key_mode: str = "luma"

    # Black/white level (0-255) used to map luminance -> alpha.
    # Pixels with luma <= black_level become 100% transparent;
    # pixels with luma >= white_level become 100% opaque; smooth ramp between.
    black_level: float = 8.0
    white_level: float = 235.0

    # "color" mode: background color to subtract (BGR), and the color-distance
    # ramp (in BGR Euclidean units, 0-441.7) that maps to alpha. Pixels within
    # key_similarity of key_color_bgr become 100% transparent; pixels farther
    # than key_similarity + key_smoothness become 100% opaque; smooth ramp
    # between.
    key_color_bgr: tuple = (0, 0, 0)
    key_similarity: float = 30.0
    key_smoothness: float = 60.0

    # Shadow offset in pixels (in the resolution of the received frame).
    shadow_dx: float = 3.0
    shadow_dy: float = 3.0
    shadow_blur_sigma: float = 4.0
    shadow_opacity: float = 0.6
    shadow_color_bgr: tuple = (0, 0, 0)

    # Solid color the keyed text is repainted with (replaces its original color).
    text_color_bgr: tuple = (255, 255, 255)

    # Optional background box behind the shadow/text, at a fixed opacity — a
    # plate to improve legibility, independent of the keyed shape. box_w/box_h
    # are in pixels (frame resolution); 0 means "full frame width/height".
    # box_pos_x/box_pos_y offset the box in pixels from the frame center,
    # using the same convention as OutputConfig: +x = right, +y = up.
    box_enabled: bool = False
    box_opacity: float = 0.5
    box_color_bgr: tuple = (0, 0, 0)
    box_w: float = 0.0
    box_h: float = 0.0
    box_pos_x: float = 0.0
    box_pos_y: float = 0.0


def _box_mask(shape, cfg):
    """Builds an HxWx1 float32 alpha mask (0 or box_opacity) for the
    background box rectangle, sized and positioned per `cfg` within a frame
    of `shape` = (H, W)."""
    h, w = shape
    box_w = w if cfg.box_w <= 0 else float(cfg.box_w)
    box_h = h if cfg.box_h <= 0 else float(cfg.box_h)
    cx, cy = w / 2.0, h / 2.0
    x0 = cx - box_w / 2.0 + cfg.box_pos_x
    y0 = cy - box_h / 2.0 - cfg.box_pos_y
    x1, y1 = x0 + box_w, y0 + box_h

    cols = np.arange(w, dtype=np.float32)
    rows = np.arange(h, dtype=np.float32)
    col_mask = (cols >= x0) & (cols < x1)
    row_mask = (rows >= y0) & (rows < y1)
    mask = (row_mask[:, None] & col_mask[None, :]).astype(np.float32)
    return mask[:, :, None] * cfg.box_opacity


def _shift(img_f32, dx, dy):
    h, w = img_f32.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        img_f32, m, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def process_frame(frame_bgrx, cfg: KeyerConfig) -> np.ndarray:
    """frame_bgrx: HxWx4 uint8 (BGRA or BGRX; input alpha channel is ignored).
    Returns HxWx4 uint8 BGRA with straight (non-premultiplied) alpha."""

    bgr = frame_bgrx[:, :, :3].astype(np.float32)

    if cfg.key_mode == "color":
        key_color = np.asarray(cfg.key_color_bgr, dtype=np.float32)
        dist = np.sqrt(np.sum((bgr - key_color[None, None, :]) ** 2, axis=2))
        span = max(1e-6, cfg.key_smoothness)
        fg_a = np.clip((dist - cfg.key_similarity) / span, 0.0, 1.0)  # HxW, 0..1
    else:
        # Perceptual luminance (Rec.601 weights, BGR order).
        luma = 0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]
        span = max(1e-6, cfg.white_level - cfg.black_level)
        fg_a = np.clip((luma - cfg.black_level) / span, 0.0, 1.0)  # HxW, 0..1

    # Shadow mask: same shape as the text, offset and blurred.
    shadow_a = _shift(fg_a, cfg.shadow_dx, cfg.shadow_dy)
    if cfg.shadow_blur_sigma > 0:
        shadow_a = cv2.GaussianBlur(shadow_a, (0, 0), cfg.shadow_blur_sigma)
    shadow_a = np.clip(shadow_a * cfg.shadow_opacity, 0.0, 1.0)

    fg_a3 = fg_a[:, :, None]
    shadow_a3 = shadow_a[:, :, None]
    text_color = np.asarray(cfg.text_color_bgr, dtype=np.float32)
    shadow_color = np.asarray(cfg.shadow_color_bgr, dtype=np.float32)
    box_color = np.asarray(cfg.box_color_bgr, dtype=np.float32)
    box_a = _box_mask(fg_a.shape, cfg) if cfg.box_enabled and cfg.box_opacity > 0 else 0.0

    # Layered "over" compositing in premultiplied space (avoids color fringing
    # at edges): background box, then shadow over the box, then text on top.
    box_a1 = shadow_a3 + box_a * (1.0 - shadow_a3)
    box_rgb1_premult = shadow_color * shadow_a3 + box_color * box_a * (1.0 - shadow_a3)

    out_a3 = fg_a3 + box_a1 * (1.0 - fg_a3)
    out_rgb_premult = text_color * fg_a3 + box_rgb1_premult * (1.0 - fg_a3)

    safe_a = np.maximum(out_a3, 1e-6)
    out_rgb = np.where(out_a3 > 1e-6, out_rgb_premult / safe_a, 0.0)

    out = np.empty_like(frame_bgrx)
    out[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(out_a3[:, :, 0] * 255.0, 0, 255).astype(np.uint8)
    return out


def place_on_canvas(bgra, canvas_w, canvas_h, scale, pos_x_px, pos_y_px):
    """Scales `bgra` (HxWx4) and places it on a transparent canvas_w x canvas_h
    canvas, offset in pixels from the canvas center: pos_x_px positive shifts
    right; pos_y_px positive shifts UP, negative shifts DOWN.
    If the scaled content is larger than the canvas, or the offset pushes it
    out, whatever doesn't fit is simply clipped — including when the content
    already matches the canvas size (the offset still works in that case)."""
    h, w = bgra.shape[:2]
    canvas_w = max(1, int(canvas_w))
    canvas_h = max(1, int(canvas_h))
    scale = max(0.01, float(scale))

    scaled_w = max(1, round(w * scale))
    scaled_h = max(1, round(h * scale))
    if scaled_w != w or scaled_h != h:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(bgra, (scaled_w, scaled_h), interpolation=interp)
    else:
        resized = bgra

    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

    off_x = round((canvas_w - scaled_w) / 2.0 + pos_x_px)
    off_y = round((canvas_h - scaled_h) / 2.0 - pos_y_px)

    src_x0 = max(0, -off_x)
    src_y0 = max(0, -off_y)
    dst_x0 = max(0, off_x)
    dst_y0 = max(0, off_y)
    copy_w = min(scaled_w - src_x0, canvas_w - dst_x0)
    copy_h = min(scaled_h - src_y0, canvas_h - dst_y0)

    if copy_w > 0 and copy_h > 0:
        canvas[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = (
            resized[src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w]
        )
    return canvas
