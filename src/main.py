import argparse
import signal
import sys
import time

import ndi_io
from compositor import FPS_PRESETS, KeyerConfig, OutputConfig, place_on_canvas, process_frame

APP_NAME = "AlphaSubs"


def parse_args():
    p = argparse.ArgumentParser(
        description="Receives an NDI subtitle signal (black background, white text), "
        "keys out the background, adds a drop shadow, and re-emits the result "
        "with alpha via NDI."
    )
    p.add_argument("--list-sources", action="store_true",
                    help="List the NDI sources available on the network and exit.")
    p.add_argument("--source", type=str, default=None,
                    help="Substring of the input NDI source name (e.g. 'OBS', 'CAPTION-PC'). "
                         "If omitted, uses the first source found.")
    p.add_argument("--output-name", type=str, default=APP_NAME,
                    help=f"Name of the output NDI source (default: '{APP_NAME}').")

    p.add_argument("--black-level", type=float, default=8.0,
                    help="Luminance (0-255) treated as pure background -> alpha 0.")
    p.add_argument("--white-level", type=float, default=235.0,
                    help="Luminance (0-255) treated as full text -> alpha 255.")

    p.add_argument("--shadow-dx", type=float, default=3.0, help="Shadow X offset (px).")
    p.add_argument("--shadow-dy", type=float, default=3.0, help="Shadow Y offset (px).")
    p.add_argument("--shadow-blur", type=float, default=4.0, help="Shadow Gaussian blur sigma.")
    p.add_argument("--shadow-opacity", type=float, default=0.6, help="Maximum shadow opacity (0-1).")
    p.add_argument("--shadow-color", type=str, default="0,0,0",
                    help="Shadow color in BGR, e.g. '0,0,0' for black.")

    p.add_argument("--text-color", type=str, default="255,255,255",
                    help="Color the keyed text is repainted with, in BGR "
                         "(default: '255,255,255', white).")

    p.add_argument("--box", action="store_true",
                    help="Enable a background box behind the shadow/text.")
    p.add_argument("--box-opacity", type=float, default=0.5,
                    help="Background box opacity (0-1).")
    p.add_argument("--box-color", type=str, default="0,0,0",
                    help="Background box color in BGR, e.g. '0,0,0' for black.")
    p.add_argument("--box-width", type=float, default=0.0,
                    help="Background box width (px). 0 = full frame width.")
    p.add_argument("--box-height", type=float, default=0.0,
                    help="Background box height (px). 0 = full frame height.")
    p.add_argument("--box-pos-x", type=float, default=0.0,
                    help="Box X offset (px) from the frame center (positive = right).")
    p.add_argument("--box-pos-y", type=float, default=0.0,
                    help="Box Y offset (px) from the frame center "
                         "(positive = up, negative = down).")

    p.add_argument("--out-width", type=int, default=1920, help="Output canvas width (px).")
    p.add_argument("--out-height", type=int, default=1080, help="Output canvas height (px).")
    p.add_argument("--out-fps", type=str, default="29.97",
                    help="Output fps as 'N/D' (e.g. '30000/1001') or a decimal (e.g. '25').")
    p.add_argument("--scale", type=float, default=1.0,
                    help="Scale of the input signal inside the output canvas.")
    p.add_argument("--pos-x", type=float, default=0.0,
                    help="X offset (px) of the signal from the canvas center "
                         "(positive = right).")
    p.add_argument("--pos-y", type=float, default=0.0,
                    help="Y offset (px) of the signal from the canvas center "
                         "(positive = up, negative = down).")

    p.add_argument("--find-timeout", type=int, default=5000,
                    help="Timeout (ms) per round of NDI source discovery.")
    p.add_argument("--recv-timeout", type=int, default=5000,
                    help="Timeout (ms) to wait for a frame before reporting 'no signal'.")
    return p.parse_args()


def parse_fps(text):
    """Accepts a known preset (e.g. '29.97', '25'), 'N/D' (e.g. '30000/1001'),
    or an arbitrary decimal (e.g. '24.5')."""
    if text in FPS_PRESETS:
        return FPS_PRESETS[text]
    if "/" in text:
        n, d = text.split("/", 1)
        return int(n), int(d)
    value = float(text)
    return round(value * 1001), 1001


def resolve_source(args):
    if args.source:
        print(f"Searching for an NDI source containing '{args.source}'...")
        src = ndi_io.find_source_by_name(args.source, timeout_ms=args.find_timeout, rounds=10)
        if src is None:
            print(f"No source found containing '{args.source}'.", file=sys.stderr)
            sys.exit(1)
        return src

    print("Searching for NDI sources on the network...")
    for _ in range(10):
        sources = ndi_io.find_sources(args.find_timeout)
        if sources:
            print(f"Using the first source found: {sources[0].ndi_name}")
            return sources[0]
    print("No NDI source found.", file=sys.stderr)
    sys.exit(1)


def main():
    args = parse_args()
    ndi_io.initialize()

    if args.list_sources:
        for src in ndi_io.find_sources(args.find_timeout):
            print(src.ndi_name)
        ndi_io.shutdown()
        return

    shadow_color = tuple(int(c) for c in args.shadow_color.split(","))
    text_color = tuple(int(c) for c in args.text_color.split(","))
    box_color = tuple(int(c) for c in args.box_color.split(","))
    cfg = KeyerConfig(
        black_level=args.black_level,
        white_level=args.white_level,
        shadow_dx=args.shadow_dx,
        shadow_dy=args.shadow_dy,
        shadow_blur_sigma=args.shadow_blur,
        shadow_opacity=args.shadow_opacity,
        shadow_color_bgr=shadow_color,
        text_color_bgr=text_color,
        box_enabled=args.box,
        box_opacity=args.box_opacity,
        box_color_bgr=box_color,
        box_w=args.box_width,
        box_h=args.box_height,
        box_pos_x=args.box_pos_x,
        box_pos_y=args.box_pos_y,
    )
    fps_n, fps_d = parse_fps(args.out_fps)
    out_cfg = OutputConfig(
        canvas_w=args.out_width,
        canvas_h=args.out_height,
        fps_n=fps_n,
        fps_d=fps_d,
        scale=args.scale,
        pos_x_px=args.pos_x,
        pos_y_px=args.pos_y,
    )

    source = resolve_source(args)
    receiver = ndi_io.Receiver(source)
    sender = ndi_io.Sender(args.output_name)
    print(f"Receiving '{source.ndi_name}' -> processing -> sending '{args.output_name}' "
          f"({out_cfg.canvas_w}x{out_cfg.canvas_h} @ {fps_n}/{fps_d} fps, NDI BGRA with alpha).")
    print("Ctrl+C to stop.")

    running = {"go": True}

    def stop(_sig, _frame):
        running["go"] = False

    signal.signal(signal.SIGINT, stop)

    frames = 0
    last_report = time.time()
    try:
        while running["go"]:
            frame = receiver.read(timeout_ms=args.recv_timeout)
            if frame is None:
                continue
            out = process_frame(frame, cfg)
            canvas = place_on_canvas(
                out, out_cfg.canvas_w, out_cfg.canvas_h,
                out_cfg.scale, out_cfg.pos_x_px, out_cfg.pos_y_px,
            )
            sender.send(canvas, frame_rate_n=out_cfg.fps_n, frame_rate_d=out_cfg.fps_d)
            frames += 1

            now = time.time()
            if now - last_report >= 5.0:
                fps = frames / (now - last_report)
                print(f"{fps:.1f} fps")
                frames = 0
                last_report = now
    finally:
        receiver.close()
        sender.close()
        ndi_io.shutdown()
        print("Stopped.")


if __name__ == "__main__":
    main()
