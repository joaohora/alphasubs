"""Tkinter GUI for the AlphaSubs NDI pipeline (cross-platform).

Runs identically on Windows and macOS. Lets you pick the input NDI source,
tune the keying/shadow parameters live, save presets, zoom the preview,
and mute the NDI output without stopping the app — all without touching
the terminal.
"""

import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import astuple
from pathlib import Path
from tkinter import colorchooser, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

import languages
import ndi_io
from captions import CaptionBoard, CaptionConfig, load_font, render_keyed_captions
from compositor import FPS_PRESETS, KeyerConfig, OutputConfig, place_on_canvas, process_frame

try:
    import audio_io
    import speech
    import voice as tts
except Exception as _e:  # e.g. PortAudio missing: the audio mode is then unavailable
    audio_io = speech = tts = None
    AUDIO_IMPORT_ERROR = str(_e)
else:
    AUDIO_IMPORT_ERROR = None

APP_NAME = "AlphaSubs"

PREVIEW_MAX_W = 480
PREVIEW_INTERVAL_S = 1.0 / 8
INFO_INTERVAL_S = 1.0
MAX_LOG_LINES = 500
PRESETS_PATH = Path.home() / ".alphasubs_presets.json"
PRESET_SLOTS = 8
LONG_PRESS_MS = 550
SAVE_FLASH_MS = 500
ZOOM_MIN, ZOOM_MAX = 1.0, 4.0
PRESET_ATTRS = (
    "black_level", "white_level", "shadow_dx", "shadow_dy",
    "shadow_blur_sigma", "shadow_opacity",
    "box_opacity", "box_w", "box_h", "box_pos_x", "box_pos_y",
    "key_similarity", "key_smoothness",
)
PRESET_COLOR_ATTRS = ("text_color_bgr", "shadow_color_bgr", "box_color_bgr", "key_color_bgr")
KEY_MODE_LABELS = {"luma": "Black background (luma)", "color": "Custom color"}
OUTPUT_ATTRS = ("canvas_w", "canvas_h", "fps_n", "fps_d", "scale", "pos_x_px", "pos_y_px")
CAPTION_ATTRS = ("font_size", "max_lines", "line_width_pct", "bottom_margin", "hold_s")

INPUT_MODE_LABELS = {"ndi": "NDI video (key)", "audio": "Audio (speech-to-text)"}
# Language choices depend on the engine (see languages.py).
AUTO_LANG_LABELS = {"basic": "Auto (EN / PT / ES)", "any": "Auto (any language)"}
SAME_LANG_LABEL = "Same as subtitles"
LANG_BY_LABEL = {languages.label(code): code for code in languages.ALL}
ENGINE_LABELS = {
    "local": "Local (Whisper, offline)", "azure": "Cloud (Azure Speech)",
    "soniox": "Cloud (Soniox)", "speechmatics": "Cloud (Speechmatics)",
}
ENGINE_SHORT = {"local": "Local", "azure": "Azure", "soniox": "Soniox", "speechmatics": "Speechmatics"}
TRANSLATION_LABELS = {"engine": "By the engine", "claude": "Claude (with context, pt-PT)"}
WHISPER_MODELS = ("auto", "large-v3-turbo", "large-v3", "medium", "small", "base")
CHANNEL_MIX_LABEL = "Mix (1+2)"
NDI_INPUT_PREFIX = "NDI: "
LEVEL_METER_W, LEVEL_METER_H = 140, 10
CARD_TEXT_W = 175  # a module card's text width (its labels wrap at 180 px)
LEVEL_INTERVAL_S = 0.1
# The app's configuration beyond the look — input, recognition, voice and the
# output name. Saved in each preset (under "_app") and, together with the
# API keys (app-wide, never in presets), as the last session.
APP_SETTING_KEYS = (
    "input_mode", "ndi_source_name", "output_name",
    "device_label", "channel", "source_lang", "target_lang", "engine", "whisper_model", "hotwords",
    "translation", "claude_model",
    "voice_engine", "voice_name", "voice_lang", "voice_device_label", "voice_ndi",
    "voice_speed", "voice_volume", "voice_buffer_ms",
)
VOICE_ENGINE_LABELS = {
    "off": "Off", "local": "Local (Piper, offline)", "azure": "Cloud (Azure neural)", "soniox": "Cloud (Soniox)",
}
NO_DEVICE_LABEL = "None"
POS_RANGE = 1000
BOX_SIZE_MAX = 3840
BOX_POS_RANGE = 1000

RESOLUTION_PRESETS = {
    "1920x1080 (Full HD)": (1920, 1080),
    "1280x720 (HD)": (1280, 720),
    "3840x2160 (4K UHD)": (3840, 2160),
    "1080x1920 (Vertical Full HD)": (1080, 1920),
    "Custom": None,
}

# ---------- Dark "tech" palette ----------
BG = "#14161a"
PANEL_BG = "#1c1f26"
FIELD_BG = "#20242c"
BORDER = "#2c313b"
FG = "#e7e9ee"
MUTED_FG = "#8b93a3"
ACCENT = "#3ea6ff"
ACCENT_HOVER = "#5cb6ff"
ACCENT_FG = "#06121f"
PLAY_BG, PLAY_HOVER = "#1f9d55", "#27b768"
REC_BG = "#e5484d"
FONT_CANDIDATES = ("Segoe UI", "Helvetica Neue", "Helvetica", "Arial")

# Flat "broadcast console" buttons: kind -> (fill color or None for outline, text
# color, bold). Outline buttons (fill=None) get a BG/PANEL_BG/FIELD_BG tint on
# hover/press instead of a solid color; only state-carrying kinds use a fill.
BUTTON_KINDS = {
    "neutral": (None, FG, False),
    "positive": (PLAY_BG, "#ffffff", True),
    "negative": (REC_BG, "#ffffff", True),
    "accent": (ACCENT, ACCENT_FG, True),
}
BUTTON_RADIUS = 3


REPO_ROOT = Path(__file__).resolve().parent.parent
GIT_REMOTE = "origin"
GIT_BRANCH = "main"
# pip output lines worth showing in the Log while an update installs
# libraries, and how many last lines to show if it fails.
PIP_LOG_PREFIXES = ("Collecting", "Installing", "Successfully", "ERROR")
PIP_LOG_TAIL = 15


def _bundle_base_dir():
    # In a PyInstaller --onefile build, bundled data (see --add-data in
    # build_windows.bat) is unpacked next to sys._MEIPASS at runtime instead
    # of living under REPO_ROOT.
    return Path(getattr(sys, "_MEIPASS", REPO_ROOT))


def _load_version():
    try:
        path = _bundle_base_dir() / "VERSION"
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        version = lines[0] if len(lines) > 0 else "0.0.0"
        release_date = lines[1] if len(lines) > 1 else ""
        return version, release_date
    except OSError:
        return "0.0.0", ""


APP_VERSION, APP_RELEASE_DATE = _load_version()


def _icon_path():
    return _bundle_base_dir() / "assets" / "icon.png"


class App:
    def __init__(self, root):
        self.root = root
        root.title(f"{APP_NAME} (NDI)")
        try:
            self._icon_image = ImageTk.PhotoImage(Image.open(_icon_path()))
            root.iconphoto(True, self._icon_image)
        except (OSError, tk.TclError):
            pass

        self.live_cfg = KeyerConfig()
        self.sliders = {}
        self.sources = []
        self.worker = None
        self.stop_event = threading.Event()
        self.msg_queue = queue.Queue()
        self.preview_photo = None
        self.preview_window = None
        self.preview_canvas = None
        self.log_window = None
        self.log_text = None
        self.log_lines = []
        self.zoom = 1.0
        self.muted = False
        self.last_raw_bgr = None
        self.picking_color = False
        self._pick_scale = 1.0
        self.presets = self._load_presets()
        self.output_cfg = self._load_output_defaults()
        self.caption_cfg, self.audio_prefs = self._load_audio_defaults()
        self.input_mode = self.audio_prefs.get("input_mode", "ndi")
        if self.input_mode == "audio" and speech is None:
            self.input_mode = "ndi"
        self.audio_devices = []  # sound devices + NDI sources, see _rebuild_audio_inputs
        self._sound_inputs = []
        self.output_devices = []
        self.voice_tuning = (
            tts.VoiceTuning(
                speed=float(self.audio_prefs.get("voice_speed", 1.0)),
                volume=float(self.audio_prefs.get("voice_volume", 1.0)),
                buffer_ms=float(self.audio_prefs.get("voice_buffer_ms", tts.BUFFER_MS_DEFAULT)),
            )
            if tts else None
        )
        self._lock_while_running = []  # (widget, state to restore) for input settings
        self._monitor = None  # meter-only audio capture while stopped
        self._monitor_gen = 0
        self._listening = False  # the monitor's audio also plays on the default output
        self._listen_out = None
        self._listen_btns = []
        self._engine = None  # the running speech engine (audio mode), for listening
        # Listening to the translated voice (VOICE card), while it runs.
        self._voice_listening = False
        self._voice_listen_out = None
        self._voice_listen_btns = []
        self._voice_out = None  # the running VoiceOutput
        # NDI (key mode) source from a preset / the last session, selected
        # once discovery finds it.
        self._wanted_ndi_source = self.audio_prefs.get("ndi_source_name") or None
        # A preset's setup applied while running, waiting for Stop.
        self._pending_app_settings = None

        ndi_io.initialize()
        self._source_finder = ndi_io.Finder()
        self._sources_stop_event = threading.Event()
        self._sources_thread = threading.Thread(
            target=self._sources_worker_loop, daemon=True
        )

        self._apply_theme()
        self._build_scroll_container()
        self._build_ui()
        self._lock_window_size()
        self._sources_thread.start()
        if (REPO_ROOT / ".git").is_dir():
            threading.Thread(target=self._startup_requirements_worker, daemon=True).start()
        self.root.after(80, self._poll_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<KeyPress-m>", self._toggle_mute)
        self.root.bind("<KeyPress-M>", self._toggle_mute)

    # ---------- Theme ----------

    def _apply_theme(self):
        root = self.root
        root.configure(bg=BG)
        self._darken_titlebar(root)

        try:
            families = set(tkfont.families(root))
        except tk.TclError:
            families = set()
        self.font_family = next((f for f in FONT_CANDIDATES if f in families), "TkDefaultFont")
        base_font = (self.font_family, 10)
        bold_font = (self.font_family, 10, "bold")
        small_bold_font = (self.font_family, 9, "bold")
        self.bold_font = bold_font
        self.normal_font = base_font

        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                tkfont.nametofont(name).configure(family=self.font_family, size=10)
            except tk.TclError:
                pass

        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=BG, foreground=FG, font=base_font)
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG, font=base_font)
        style.configure(
            "TLabelframe", background=BG, bordercolor=BORDER, borderwidth=1, relief="solid"
        )
        style.configure("TLabelframe.Label", background=BG, foreground=MUTED_FG, font=small_bold_font)
        style.configure(
            "TEntry", fieldbackground=FIELD_BG, foreground=FG, bordercolor=BORDER,
            insertcolor=FG, padding=4,
        )
        style.map("TEntry", fieldbackground=[("readonly", FIELD_BG)])
        style.configure(
            "TCombobox", fieldbackground=FIELD_BG, foreground=FG, background=FIELD_BG,
            arrowcolor=MUTED_FG, bordercolor=BORDER, padding=4,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", FIELD_BG)],
            foreground=[("readonly", FG)],
        )
        style.configure(
            "Horizontal.TScale", background=BG, troughcolor=FIELD_BG,
            bordercolor=BG, lightcolor=ACCENT, darkcolor=ACCENT,
        )
        style.configure("TSeparator", background=BORDER)
        style.configure(
            "TCheckbutton", background=BG, foreground=FG, font=base_font,
            indicatorbackground=FIELD_BG, indicatorforeground=FG,
        )
        style.map(
            "TCheckbutton",
            background=[("active", BG)],
            indicatorbackground=[("selected", ACCENT), ("active", FIELD_BG)],
        )
        style.configure(
            "TScrollbar", background=PANEL_BG, troughcolor=BG,
            bordercolor=BORDER, arrowcolor=MUTED_FG,
        )

        root.option_add("*TCombobox*Listbox.background", FIELD_BG)
        root.option_add("*TCombobox*Listbox.foreground", FG)
        root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        root.option_add("*TCombobox*Listbox.selectForeground", ACCENT_FG)
        root.option_add("*Font", base_font)

    def _build_scroll_container(self):
        """Wraps all main-window content in a canvas + vertical scrollbar, so
        the app stays usable on small/low-resolution displays where the full
        content wouldn't otherwise fit on screen. `self.content` is the frame
        that all UI sections should be built into (instead of `self.root`)."""
        canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        vscroll = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        content = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")

        def on_content_configure(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event):
            canvas.itemconfig(window_id, width=event.width)

        content.bind("<Configure>", on_content_configure)
        canvas.bind("<Configure>", on_canvas_configure)

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        self._scroll_canvas = canvas
        self._vscroll = vscroll
        self.content = content

    def _lock_window_size(self):
        """Sizes the main window to fit its content, but capped to the screen
        size (leaving room for the taskbar/title bar). On small displays the
        window shrinks to fit and the content becomes scrollable instead of
        being cut off. No manual resizing (no drag-resize/maximize)."""
        root = self.root
        root.update_idletasks()
        content_w = self.content.winfo_reqwidth()
        content_h = self.content.winfo_reqheight()
        sb_w = self._vscroll.winfo_reqwidth()

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        max_h = max(300, screen_h - 90)
        win_h = min(content_h, max_h)
        win_w = min(content_w + sb_w, screen_w - 40)

        root.geometry(f"{win_w}x{win_h}")
        root.resizable(False, False)

    # ---------- UI ----------

    def _build_ui(self):
        pad = {"padx": 10, "pady": 4}

        self.input_info_var = tk.StringVar(value="Input: —")
        self.output_info_var = tk.StringVar(value="Output: —")
        self.box_enabled_var = tk.BooleanVar(value=self.live_cfg.box_enabled)

        self._build_pipeline_row(pad)
        self._build_presets_ui(pad)
        ttk.Separator(self.content, orient="horizontal").pack(fill="x", padx=pad["padx"], pady=6)

        controls = ttk.Frame(self.content)
        controls.pack(fill="x", padx=pad["padx"], pady=(0, 4))
        self.start_btn = self._make_float_button(
            controls,
            {"idle": ("Start", "positive"), "running": ("Stop", "negative")},
            "idle", command=self._toggle_start_stop,
        )
        self.start_btn.pack(side="left")

        self.mute_btn = self._make_float_button(
            controls,
            {"idle": ("Mute (M)", "neutral"), "muted": ("Muted (M)", "accent")},
            "idle", command=self._toggle_mute,
        )
        self.mute_btn.pack(side="left", padx=(18, 6))

        self._make_simple_button(controls, "Preview...", self._open_preview_window).pack(
            side="left", padx=(18, 6)
        )
        self._make_simple_button(controls, "Log...", self._open_log_window).pack(side="left")

        self.status_var = tk.StringVar(value="Stopped.")
        ttk.Label(controls, textvariable=self.status_var, foreground=MUTED_FG).pack(side="left", padx=12)

        if (REPO_ROOT / ".git").is_dir():
            self._make_simple_button(
                controls, "Check for Updates", self._check_for_updates
            ).pack(side="right")

        footer = ttk.Frame(self.content)
        footer.pack(fill="x", padx=pad["padx"], pady=(0, 8))
        version_text = f"{APP_NAME} v{APP_VERSION}"
        if APP_RELEASE_DATE:
            version_text += f"  ·  {APP_RELEASE_DATE}"
        ttk.Label(
            footer, text=version_text, foreground=MUTED_FG, font=(self.font_family, 8)
        ).pack()

        # Each pipeline module's full specs live in its own floating, non-modal
        # panel (built once, shown/hidden — never destroyed, so their widgets
        # stay valid even while the panel is hidden).
        self._build_source_panel()
        self._build_text_panel()
        self._build_background_panel()
        self._build_output_panel()
        self._build_voice_panel()

        self._update_output_info_label()
        self._update_source_card()
        self._update_text_card()
        self._update_background_card()
        self._update_output_card()
        self._update_voice_card()

    # ---------- Pipeline row (home screen) ----------

    def _build_pipeline_row(self, pad):
        """The home screen: the signal chain as a row of clickable module
        cards connected by arrows. Each card shows a glanceable summary of
        that module's current specs; clicking it opens its floating panel."""
        row = ttk.Frame(self.content)
        row.pack(fill="x", padx=pad["padx"], pady=(10, 6))

        self.source_card_var = tk.StringVar(value="No source selected")
        self.source_title_var = tk.StringVar(value="NDI SOURCE")
        self.text_card_var = tk.StringVar(value="")
        self.background_card_var = tk.StringVar(value="")
        self.output_card_var = tk.StringVar(value="")

        self._make_module_card(
            row, self.source_title_var, self.source_card_var, self._open_source_panel,
            extra=self._add_source_card_meter,
        ).pack(side="left")
        self._make_arrow(row).pack(side="left", padx=4)
        self._make_module_card(row, "TEXT", self.text_card_var, self._open_text_panel).pack(side="left")
        self._make_arrow(row).pack(side="left", padx=4)
        self._make_module_card(
            row, "BACKGROUND", self.background_card_var, self._open_background_panel,
            extra=self._add_background_card_checkbox,
        ).pack(side="left")
        self._make_arrow(row).pack(side="left", padx=4)
        self._make_module_card(row, "OUTPUT", self.output_card_var, self._open_output_panel).pack(side="left")
        # The translated voice branches off the chain (it runs alongside the
        # captions, from the same recognized text), hence "+" not an arrow.
        self.voice_card_var = tk.StringVar(value="")
        tk.Label(row, text="+", bg=BG, fg=MUTED_FG, font=(self.font_family, 18, "bold")).pack(side="left", padx=4)
        self._make_module_card(
            row, "VOICE", self.voice_card_var, self._open_voice_panel, extra=self._add_voice_card_listen,
        ).pack(side="left")

    def _add_voice_card_listen(self, inner):
        """Extra content for the VOICE card: the listen (speaker) toggle, to
        hear the translated voice on this computer while it runs."""
        self.voice_listen_row = tk.Frame(inner, bg=PANEL_BG)
        self._make_listen_button(self.voice_listen_row, 16, bg=PANEL_BG, voice=True).pack(side="left")

    def _add_source_card_meter(self, inner):
        """Extra content for the input card: a small input level meter and
        the listen (speaker) toggle, shown in the audio mode so the signal
        can be checked at a glance — and heard — before and after Start."""
        self.card_level_row = tk.Frame(inner, bg=PANEL_BG)
        self.card_level_canvas = tk.Canvas(
            self.card_level_row, width=LEVEL_METER_W, height=6, bg=FIELD_BG,
            highlightthickness=1, highlightbackground=BORDER,
        )
        self.card_level_canvas.pack(side="left")
        self.card_level_bar = self.card_level_canvas.create_rectangle(0, 0, 0, 8, fill=PLAY_BG, width=0)
        self._make_listen_button(self.card_level_row, 16, bg=PANEL_BG).pack(side="left", padx=(6, 0))

    def _add_background_card_checkbox(self, inner):
        """Extra content for the BACKGROUND card: the Enabled checkbox, right
        on the card, so it can be toggled without opening the full panel."""
        tk.Checkbutton(
            inner, text="Enabled", variable=self.box_enabled_var, command=self._toggle_box_enabled,
            bg=PANEL_BG, fg=FG, activebackground=PANEL_BG, activeforeground=FG,
            selectcolor=FIELD_BG, highlightthickness=0, bd=0, font=self.normal_font,
        ).pack(anchor="w", pady=(2, 0))

    def _make_arrow(self, parent):
        return tk.Label(parent, text="→", bg=BG, fg=MUTED_FG, font=(self.font_family, 18, "bold"))

    def _make_module_card(self, parent, title, info_var, command, extra=None):
        """A clickable card (title + live summary) representing one pipeline
        module. Reused, fixed-size flat panel-style box, matching the app's
        broadcast-console look; hover highlights its border in accent color.
        Its content is vertically centered via an inner frame that packs with
        expand=True. `extra`, if given, is called with that inner frame so a
        card can host a small interactive control (e.g. an inline checkbox)
        in addition to its title/summary — that control keeps its own click
        behavior and doesn't open the panel. `title` may be a StringVar, for
        a card whose title follows the current mode."""
        card = tk.Frame(
            parent, bg=PANEL_BG, highlightthickness=1, highlightbackground=BORDER,
            cursor="hand2", width=200, height=92,
        )
        card.pack_propagate(False)

        inner = tk.Frame(card, bg=PANEL_BG)
        inner.pack(expand=True)

        title_kw = {"textvariable": title} if isinstance(title, tk.StringVar) else {"text": title}
        title_lbl = tk.Label(
            inner, bg=PANEL_BG, fg=MUTED_FG, font=(self.font_family, 9, "bold"), anchor="w", **title_kw,
        )
        title_lbl.pack(fill="x", pady=(0, 3), anchor="w")

        info_lbl = tk.Label(
            inner, textvariable=info_var, bg=PANEL_BG, fg=FG, font=self.normal_font,
            anchor="w", justify="left", wraplength=180,
        )
        info_lbl.pack(fill="x", anchor="w")

        if extra:
            extra(inner)

        def on_enter(_event):
            card.config(highlightbackground=ACCENT)

        def on_leave(_event):
            card.config(highlightbackground=BORDER)

        def on_click(_event):
            command()

        for w in (card, inner, title_lbl, info_lbl):
            w.bind("<Enter>", on_enter)
            w.bind("<Leave>", on_leave)
            w.bind("<Button-1>", on_click)

        return card

    def _update_source_card(self):
        if self.input_mode == "audio":
            self.card_level_row.pack(anchor="w", pady=(4, 0))
        else:
            self.card_level_row.pack_forget()
        if self.input_mode == "audio":
            self.source_title_var.set("AUDIO INPUT")
            # Cut to what fits on one line of the card (by width, not
            # characters): a wrapped name would push the level meter and the
            # listen button out of the fixed-height card.
            device = self.audio_device_var.get().strip() or "No device selected"
            font = tkfont.Font(font=self.normal_font)
            if font.measure(device) > CARD_TEXT_W:
                while device and font.measure(device + "…") > CARD_TEXT_W:
                    device = device[:-1]
                device = device.rstrip() + "…"
            src = self._source_lang_code().upper()
            engine = ENGINE_SHORT[self._engine_code()]
            if self._translation_code() == "claude":
                engine += " + Claude"
            self.source_card_var.set(f"{device}\n{src} → {self._target_lang_code().upper()} · {engine}")
            return
        self.source_title_var.set("NDI SOURCE")
        name = self.source_var.get().strip() or "No source selected"
        info = self.input_info_var.get()
        if info and info != "Input: —":
            self.source_card_var.set(f"{name}\n{info.replace('Input: ', '')}")
        else:
            self.source_card_var.set(name)

    def _update_text_card(self):
        if self.input_mode == "audio":
            cap = self.caption_cfg
            lines = int(cap.max_lines)
            self.text_card_var.set(
                f"Captions · {cap.font_size:.0f} px\n{lines} line{'s' if lines != 1 else ''}"
            )
        elif self.live_cfg.key_mode == "color":
            b, g, r = self.live_cfg.key_color_bgr
            self.text_card_var.set(f"Color key\nRGB({r}, {g}, {b})")
        else:
            self.text_card_var.set(
                f"Luma\n{self.live_cfg.black_level:.0f} – {self.live_cfg.white_level:.0f}"
            )

    def _update_background_card(self):
        self.background_card_var.set(f"{self.live_cfg.box_opacity * 100:.0f}% opacity")

    def _update_output_card(self):
        cfg = self.output_cfg
        fps_s = self._fmt_fps(cfg.fps_n, cfg.fps_d)
        name = self.output_name_var.get().strip() or APP_NAME if hasattr(self, "output_name_var") else APP_NAME
        self.output_card_var.set(f"{cfg.canvas_w}x{cfg.canvas_h} @ {fps_s} fps\n{name}")

    # ---------- Module panels (floating, non-modal) ----------

    def _make_panel_window(self, title):
        """A floating panel window for one pipeline module: built once and
        hidden with withdraw() (never destroyed), so it can be reopened
        instantly and its widgets stay valid — and non-modal, so the operator
        can keep watching the live Preview while adjusting it."""
        win = tk.Toplevel(self.root)
        win.withdraw()
        win.title(title)
        win.configure(bg=BG)
        self._darken_titlebar(win)
        win.protocol("WM_DELETE_WINDOW", win.withdraw)
        return win

    def _open_source_panel(self):
        self.source_panel.deiconify()
        self.source_panel.lift()

    def _open_text_panel(self):
        self.text_panel.deiconify()
        self.text_panel.lift()

    def _open_background_panel(self):
        self.background_panel.deiconify()
        self.background_panel.lift()

    def _open_output_panel(self):
        self.output_panel.deiconify()
        self.output_panel.lift()

    def _build_source_panel(self):
        win = self._make_panel_window("Input")
        self.source_panel = win

        outer = ttk.Frame(win)
        outer.pack(fill="both", expand=True, padx=10, pady=10)
        outer.columnconfigure(0, weight=1)

        mode_row = ttk.Frame(outer)
        mode_row.grid(row=0, column=0, sticky="we", pady=(0, 8))
        ttk.Label(mode_row, text="Input type:").pack(side="left")
        self.input_mode_var = tk.StringVar(value=INPUT_MODE_LABELS[self.input_mode])
        self.input_mode_combo = ttk.Combobox(
            mode_row, textvariable=self.input_mode_var, state="readonly",
            values=list(INPUT_MODE_LABELS.values()), width=24,
        )
        self.input_mode_combo.pack(side="left", padx=6)
        self.input_mode_combo.bind("<<ComboboxSelected>>", self._on_input_mode_change)
        self._lock_while_running.append((self.input_mode_combo, "readonly"))

        body = ttk.Frame(outer)
        body.grid(row=1, column=0, sticky="we")
        body.columnconfigure(1, weight=1)
        self.ndi_input_frame = body

        ttk.Label(body, text="NDI source:").grid(row=0, column=0, sticky="w")
        self.source_var = tk.StringVar(value=self._wanted_ndi_source or "")
        self.source_combo = ttk.Combobox(body, textvariable=self.source_var, state="readonly", width=42)
        self.source_combo.grid(row=0, column=1, sticky="we", padx=4)
        self.source_combo.bind("<<ComboboxSelected>>", self._on_ndi_source_selected)
        self._make_simple_button(body, "Refresh", self._refresh_sources).grid(row=0, column=2, padx=4)

        ttk.Label(body, textvariable=self.input_info_var, foreground=MUTED_FG).grid(
            row=1, column=1, sticky="w", padx=4, pady=(4, 0)
        )

        self._build_audio_input_frame(outer)
        self._show_input_mode_frames()

    def _on_ndi_source_selected(self, event=None):
        self._wanted_ndi_source = None  # the operator's pick wins over a saved one
        self._update_source_card()

    def _build_audio_input_frame(self, outer):
        """Audio (speech-to-text) mode settings: capture device/channel,
        spoken and subtitle languages, recognition engine and its options."""
        prefs = self.audio_prefs
        frame = ttk.Frame(outer)
        frame.grid(row=1, column=0, sticky="we")
        frame.columnconfigure(1, weight=1)
        self.audio_input_frame = frame

        def combo(parent, var, values, width, on_change=None):
            c = ttk.Combobox(parent, textvariable=var, state="readonly", values=values, width=width)
            c.bind("<<ComboboxSelected>>", on_change or (lambda e: self._update_source_card()))
            self._lock_while_running.append((c, "readonly"))
            return c

        def entry(parent, var, **kw):
            e = ttk.Entry(parent, textvariable=var, **kw)
            self._lock_while_running.append((e, "normal"))
            return e

        ttk.Label(frame, text="Audio device:").grid(row=0, column=0, sticky="w", pady=2)
        self.audio_device_var = tk.StringVar(value=prefs.get("device_label", ""))
        self.audio_device_combo = combo(frame, self.audio_device_var, [], 42, self._on_audio_device_change)
        self.audio_device_combo.grid(row=0, column=1, sticky="we", padx=4, pady=2)
        refresh_btn = self._make_simple_button(frame, "Refresh", lambda: self._refresh_audio_devices(rescan=True))
        refresh_btn.grid(row=0, column=2, padx=4)

        ch_row = ttk.Frame(frame)
        ch_row.grid(row=1, column=1, sticky="we", padx=4, pady=2)
        ttk.Label(frame, text="Channel:").grid(row=1, column=0, sticky="w", pady=2)
        self.audio_channel_var = tk.StringVar(value=prefs.get("channel", CHANNEL_MIX_LABEL))
        self.audio_channel_combo = combo(
            ch_row, self.audio_channel_var, [CHANNEL_MIX_LABEL], 10, lambda e: self._restart_monitor()
        )
        self.audio_channel_combo.pack(side="left")
        ttk.Label(ch_row, text="   Level:").pack(side="left")
        self.level_canvas = tk.Canvas(
            ch_row, width=LEVEL_METER_W, height=LEVEL_METER_H, bg=FIELD_BG,
            highlightthickness=1, highlightbackground=BORDER,
        )
        self.level_canvas.pack(side="left", padx=6)
        self.level_bar = self.level_canvas.create_rectangle(0, 0, 0, LEVEL_METER_H + 2, fill=PLAY_BG, width=0)
        self.level_text_var = tk.StringVar(value="— dB")
        ttk.Label(ch_row, textvariable=self.level_text_var, foreground=MUTED_FG, width=8).pack(side="left")
        self._make_listen_button(ch_row, 22).pack(side="left")

        lang_row = ttk.Frame(frame)
        lang_row.grid(row=2, column=1, columnspan=2, sticky="we", padx=4, pady=2)
        ttk.Label(frame, text="Spoken language:").grid(row=2, column=0, sticky="w", pady=2)
        # The choices are filled in by _refresh_lang_choices, per engine.
        self.source_lang_var = tk.StringVar(value=self._lang_label(prefs.get("source_lang", "auto")))
        self.source_lang_combo = combo(lang_row, self.source_lang_var, [], 18)
        self.source_lang_combo.pack(side="left")
        ttk.Label(lang_row, text="   Subtitles in:").pack(side="left")
        self.target_lang_var = tk.StringVar(value=self._lang_label(prefs.get("target_lang", "pt")))
        self.target_lang_combo = combo(lang_row, self.target_lang_var, [], 12, self._on_target_lang_change)
        self.target_lang_combo.pack(side="left", padx=6)

        ttk.Label(frame, text="Engine:").grid(row=3, column=0, sticky="w", pady=2)
        self.engine_var = tk.StringVar(value=ENGINE_LABELS.get(prefs.get("engine"), ENGINE_LABELS["local"]))
        combo(frame, self.engine_var, list(ENGINE_LABELS.values()), 24, self._on_engine_change).grid(
            row=3, column=1, sticky="w", padx=4, pady=2
        )

        self.local_engine_frame = ttk.Frame(frame)
        self.local_engine_frame.grid(row=4, column=0, columnspan=3, sticky="we", pady=2)
        ttk.Label(self.local_engine_frame, text="Whisper model:").pack(side="left")
        self.whisper_model_var = tk.StringVar(value=prefs.get("whisper_model", "auto"))
        combo(self.local_engine_frame, self.whisper_model_var, list(WHISPER_MODELS), 14).pack(side="left", padx=6)
        ttk.Label(
            self.local_engine_frame, text="auto = large-v3-turbo on NVIDIA GPU, base on CPU (small: better, slower)",
            foreground=MUTED_FG,
        ).pack(side="left", padx=6)

        self.azure_engine_frame = ttk.Frame(frame)
        self.azure_engine_frame.grid(row=4, column=0, columnspan=3, sticky="we", pady=2)
        ttk.Label(self.azure_engine_frame, text="Azure key:").pack(side="left")
        self.azure_key_var = tk.StringVar(value=prefs.get("azure_key", ""))
        entry(self.azure_engine_frame, self.azure_key_var, show="•", width=28).pack(side="left", padx=6)
        ttk.Label(self.azure_engine_frame, text="Region:").pack(side="left")
        self.azure_region_var = tk.StringVar(value=prefs.get("azure_region", "westeurope"))
        entry(self.azure_engine_frame, self.azure_region_var, width=14).pack(side="left", padx=6)

        self.key_engine_frames = {}
        for code, service in (("soniox", "Soniox"), ("speechmatics", "Speechmatics")):
            key_frame = ttk.Frame(frame)
            key_frame.grid(row=4, column=0, columnspan=3, sticky="we", pady=2)
            ttk.Label(key_frame, text=f"{service} key:").pack(side="left")
            var = tk.StringVar(value=prefs.get(f"{code}_key", ""))
            entry(key_frame, var, show="•", width=40).pack(side="left", padx=6)
            setattr(self, f"{code}_key_var", var)
            self.key_engine_frames[code] = key_frame
        ttk.Label(
            self.key_engine_frames["speechmatics"], text="needs the spoken language (no Auto)", foreground=MUTED_FG,
        ).pack(side="left", padx=6)

        ttk.Label(frame, text="Translation:").grid(row=5, column=0, sticky="w", pady=2)
        trans_row = ttk.Frame(frame)
        trans_row.grid(row=5, column=1, columnspan=2, sticky="we", padx=4, pady=2)
        self.translation_var = tk.StringVar(
            value=TRANSLATION_LABELS.get(prefs.get("translation"), TRANSLATION_LABELS["engine"])
        )
        combo(trans_row, self.translation_var, list(TRANSLATION_LABELS.values()), 26, self._on_engine_change).pack(
            side="left"
        )
        self.claude_frame = ttk.Frame(frame)
        self.claude_frame.grid(row=6, column=0, columnspan=3, sticky="we", pady=2)
        ttk.Label(self.claude_frame, text="Claude key:").pack(side="left")
        self.claude_key_var = tk.StringVar(value=prefs.get("claude_key", ""))
        entry(self.claude_frame, self.claude_key_var, show="•", width=28).pack(side="left", padx=6)
        ttk.Label(self.claude_frame, text="Model:").pack(side="left")
        claude_models = speech.CLAUDE_MODELS if speech is not None else ("claude-opus-5-5",)
        self.claude_model_var = tk.StringVar(
            value=prefs.get("claude_model") if prefs.get("claude_model") in claude_models else claude_models[0]
        )
        combo(self.claude_frame, self.claude_model_var, list(claude_models), 18).pack(side="left", padx=6)
        ttk.Label(
            self.claude_frame, text="sentences only (no live partials)", foreground=MUTED_FG,
        ).pack(side="left", padx=6)

        ttk.Label(frame, text="Names / terms:").grid(row=7, column=0, sticky="w", pady=2)
        self.hotwords_var = tk.StringVar(value=prefs.get("hotwords", ""))
        entry(frame, self.hotwords_var).grid(row=7, column=1, columnspan=2, sticky="we", padx=4, pady=2)
        ttk.Label(
            frame, text="Comma-separated, e.g. Gulbenkian, Coro Gulbenkian — helps recognize names.",
            foreground=MUTED_FG,
        ).grid(row=8, column=1, columnspan=2, sticky="w", padx=4)

        if speech is None:
            ttk.Label(
                frame, text=f"Audio mode unavailable: {AUDIO_IMPORT_ERROR}", foreground=REC_BG,
            ).grid(row=9, column=0, columnspan=3, sticky="w", pady=(6, 0))
        else:
            self._refresh_audio_devices()
        self._show_engine_frames()
        self._refresh_lang_choices()

    # ---------- Input mode (NDI video / audio) ----------

    @staticmethod
    def _lang_label(code):
        if code == "auto":
            return AUTO_LANG_LABELS["basic"]
        if code == "same":
            return SAME_LANG_LABEL
        return languages.label(code)

    @staticmethod
    def _lang_code(label, default):
        if label in AUTO_LANG_LABELS.values():
            return "auto"
        if label == SAME_LANG_LABEL:
            return "same"
        return LANG_BY_LABEL.get(label, default)

    def _source_lang_code(self):
        return self._lang_code(self.source_lang_var.get(), "auto")

    def _target_lang_code(self):
        return self._lang_code(self.target_lang_var.get(), "pt")

    def _refresh_lang_choices(self):
        """Offers the spoken and subtitle languages the engine (and the
        translation) supports; a choice no longer available falls back to
        Auto / Portuguese."""
        engine = self._engine_code()
        auto = AUTO_LANG_LABELS["any" if engine == "soniox" else "basic"]
        spoken = languages.ordered(languages.spoken(engine))
        self.source_lang_combo["values"] = [auto] + [languages.label(c) for c in spoken]
        source = self._source_lang_code()
        self.source_lang_var.set(languages.label(source) if source in spoken else auto)
        targets = languages.ordered(languages.subtitles(engine, self._translation_code()))
        self.target_lang_combo["values"] = [languages.label(c) for c in targets]
        target = self._target_lang_code()
        self.target_lang_var.set(languages.label(target if target in targets else "pt"))
        self._on_target_lang_change()

    def _engine_code(self):
        return next((k for k, v in ENGINE_LABELS.items() if v == self.engine_var.get()), "local")

    def _translation_code(self):
        return next((k for k, v in TRANSLATION_LABELS.items() if v == self.translation_var.get()), "engine")

    def _on_input_mode_change(self, event=None):
        mode = next((k for k, v in INPUT_MODE_LABELS.items() if v == self.input_mode_var.get()), "ndi")
        if mode == "audio" and speech is None:
            messagebox.showerror(APP_NAME, f"The audio mode is unavailable:\n\n{AUDIO_IMPORT_ERROR}")
            self.input_mode_var.set(INPUT_MODE_LABELS["ndi"])
            return
        self.input_mode = mode
        self._show_input_mode_frames()
        self._restart_monitor()

    def _show_input_mode_frames(self):
        """Shows the input settings and TEXT panel sections that apply to
        the current input mode (keying only applies to NDI video; caption
        layout only to audio)."""
        audio = self.input_mode == "audio"
        if audio:
            self.ndi_input_frame.grid_remove()
            self.audio_input_frame.grid()
        else:
            self.audio_input_frame.grid_remove()
            self.ndi_input_frame.grid()
        if hasattr(self, "key_settings_frame"):
            if audio:
                if self.picking_color:
                    self._end_color_pick()
                self.key_settings_frame.grid_remove()
                self.captions_frame.grid()
            else:
                self.captions_frame.grid_remove()
                self.key_settings_frame.grid()
        if hasattr(self, "source_card_var"):
            self._update_source_card()
            self._update_text_card()
        if hasattr(self, "voice_engine_var"):
            self._update_voice_card()

    def _on_engine_change(self, event=None):
        self._show_engine_frames()
        self._refresh_lang_choices()

    def _show_engine_frames(self):
        code = self._engine_code()
        frames = {"local": self.local_engine_frame, "azure": self.azure_engine_frame, **self.key_engine_frames}
        for key, frame in frames.items():
            if key == code:
                frame.grid()
            else:
                frame.grid_remove()
        if self._translation_code() == "claude":
            self.claude_frame.grid()
        else:
            self.claude_frame.grid_remove()

    def _refresh_audio_devices(self, rescan=False):
        if speech is None:
            return
        if rescan and self.worker and self.worker.is_alive():
            self._append_log("Stop the pipeline before rescanning audio devices.")
            return
        if rescan:
            self._stop_monitor()  # rescanning restarts PortAudio under open streams
        try:
            self._sound_inputs = audio_io.list_input_devices(rescan=rescan)
        except Exception as e:
            self._append_log(f"Could not list audio devices: {e}")
            self._sound_inputs = []
        self._rebuild_audio_inputs()
        current = self.audio_device_var.get()
        labels = [label for _ref, label, _ch in self.audio_devices]
        if current not in labels and not current.startswith(NDI_INPUT_PREFIX):
            default_idx = audio_io.default_input_device()
            default = next((lbl for ref, lbl, _ch in self.audio_devices if ref == default_idx), None)
            self.audio_device_var.set(default or (labels[0] if labels else ""))
        if rescan:
            self._append_log(f"{len(self._sound_inputs)} audio input device(s) found.")
        self._on_audio_device_change()
        if hasattr(self, "voice_device_combo"):
            self._refresh_output_devices()

    def _rebuild_audio_inputs(self):
        """The audio input list: sound devices, then the NDI sources on the
        network (their audio is received directly, no virtual device
        needed). Entries are (ref, label, max channels), ref being a sound
        device index or an NDI source."""
        ndi_inputs = [
            (src, f"{NDI_INPUT_PREFIX}{src.ndi_name}", audio_io.NDI_MAX_CHANNELS) for src in self.sources
        ]
        self.audio_devices = self._sound_inputs + ndi_inputs
        self.audio_device_combo["values"] = [label for _ref, label, _ch in self.audio_devices]

    def _on_sources_changed(self):
        """NDI discovery found/lost sources: refresh the NDI entries of the
        audio input list. A saved NDI selection that wasn't on the network
        yet (e.g. at startup) starts its level meter once it shows up."""
        if speech is None:
            return
        was_resolved = self._selected_audio_device()[0] is not None
        self._rebuild_audio_inputs()
        if not was_resolved and self._selected_audio_device()[0] is not None:
            self._on_audio_device_change()

    def _selected_audio_device(self):
        label = self.audio_device_var.get()
        return next(((ref, ch) for ref, lbl, ch in self.audio_devices if lbl == label), (None, 0))

    def _open_audio_input(self, meter_only):
        """A capture of the selected input (sound device or NDI source)."""
        ref, _max_ch = self._selected_audio_device()
        channel = self.audio_channel_var.get()
        channel = None if channel == CHANNEL_MIX_LABEL else int(channel) - 1
        if isinstance(ref, int):
            return audio_io.AudioCapture(ref, channel, meter_only=meter_only)
        return audio_io.NDIAudioCapture(ref, channel, meter_only=meter_only)

    def _on_audio_device_change(self, event=None):
        _ref, max_ch = self._selected_audio_device()
        values = [CHANNEL_MIX_LABEL] + [str(i) for i in range(1, max_ch + 1)]
        self.audio_channel_combo["values"] = values
        if self.audio_channel_var.get() not in values:
            self.audio_channel_var.set(CHANNEL_MIX_LABEL)
        self._update_source_card()
        self._restart_monitor()

    # ---------- Input level monitor (before Start) ----------

    def _restart_monitor(self):
        """While the audio mode is selected and stopped, keeps a meter-only
        capture of the selected device/channel open, so the operator can
        check the signal before pressing Start. While running, the meter
        follows the engine's own capture instead (and so does listening, see
        _sync_engine_listen) — the monitor is left alone."""
        if self.worker and self.worker.is_alive():
            return
        self._stop_monitor()
        if audio_io is None or self.input_mode != "audio":
            return
        if self._selected_audio_device()[0] is None:
            return
        try:
            self._monitor = self._open_audio_input(meter_only=True)
            self._monitor.start()
        except Exception as e:
            self._monitor = None
            self._append_log(f"Could not open '{self.audio_device_var.get()}' for the level meter: {e}")
            return
        if self._listening:
            self._listen_out = self._open_listen_output()
            self._monitor.listener = self._listen_out
        self._monitor_gen += 1
        self._poll_monitor(self._monitor_gen)

    def _stop_monitor(self):
        monitor, self._monitor = getattr(self, "_monitor", None), None
        if monitor is not None:
            monitor.listener = None
            try:
                monitor.close()
            except Exception:
                pass
        self._close_listen_output()
        if hasattr(self, "level_canvas"):
            self._update_level_meter(-120.0)

    def _open_listen_output(self):
        try:
            out = audio_io.ListenOutput()
            out.start()
            return out
        except Exception as e:
            self._set_listening(False)
            self._append_log(f"Could not open the audio output to listen to the input: {e}")
            return None

    def _close_listen_output(self):
        out, self._listen_out = getattr(self, "_listen_out", None), None
        if out is not None:
            try:
                out.close()
            except Exception:
                pass

    def _sync_engine_listen(self):
        """While running, listening taps the speech engine's own capture
        (there is a single capture of the device then)."""
        engine = self._engine
        if engine is None:
            return
        if self._listening:
            if self._listen_out is None:
                self._listen_out = self._open_listen_output()
            engine.set_listener(self._listen_out)
        else:
            engine.set_listener(None)
            self._close_listen_output()

    def _toggle_listen(self):
        """Speaker button: plays the selected input (device/channel, or NDI
        source audio) on the computer's default output, to hear what comes
        in — before Start (level monitor) and while running (the engine's
        capture). It stays on across Start/Stop."""
        running = bool(self.worker and self.worker.is_alive())
        turning_on = not self._listening
        self._set_listening(turning_on)
        if running:
            self._sync_engine_listen()
        else:
            self._restart_monitor()
        if not turning_on:
            self._append_log("Stopped listening to the input.")
        elif self._listen_out is not None:
            self._append_log(
                f"Listening to the input on '{self._listen_out.name}'. "
                "With a microphone, use headphones to avoid feedback."
            )
        elif self._listening and not running:  # (if not listening, the output failed and said why)
            self._set_listening(False)
            self._append_log("Nothing to listen to: select an audio input first.")

    def _toggle_voice_listen(self):
        """VOICE card speaker: plays the translated voice on the computer's
        default output while the pipeline runs (in addition to its own
        device/NDI outputs). It stays on across Start/Stop."""
        self._voice_listening = not self._voice_listening
        for btn in self._voice_listen_btns:
            btn.set_variant("on" if self._voice_listening else "off")
        running = bool(self.worker and self.worker.is_alive())
        self._sync_voice_listen()
        if not self._voice_listening:
            self._append_log("Stopped listening to the voice.")
        elif self._voice_listen_out is not None:
            self._append_log(f"Listening to the translated voice on '{self._voice_listen_out.name}'.")
        elif running:
            self._append_log("The voice isn't running — set it up (VOICE panel) and press Start again.")
        else:
            self._append_log("Listening to the voice: it will be heard here after Start.")

    def _sync_voice_listen(self):
        voice_out = self._voice_out
        if voice_out is None:
            return
        if self._voice_listening:
            if self._voice_listen_out is None:
                try:
                    self._voice_listen_out = audio_io.ListenOutput(max_delay_s=None)
                    self._voice_listen_out.start()
                except Exception as e:
                    self._voice_listen_out = None
                    self._append_log(f"Could not open the audio output to listen to the voice: {e}")
            voice_out.set_listener(self._voice_listen_out)
        else:
            voice_out.set_listener(None)
            self._close_voice_listen_output()

    def _close_voice_listen_output(self):
        out, self._voice_listen_out = self._voice_listen_out, None
        if out is not None:
            try:
                out.close()
            except Exception:
                pass

    def _set_listening(self, on):
        self._listening = on
        for btn in self._listen_btns:
            btn.set_variant("on" if on else "off")

    def _poll_monitor(self, gen):
        # `gen` ends the polling chain of a monitor that has been replaced.
        if self._monitor is None or gen != self._monitor_gen:
            return
        self._update_level_meter(self._monitor.level_db)
        self.root.after(int(LEVEL_INTERVAL_S * 1000), self._poll_monitor, gen)

    def _speech_settings(self):
        ref, _max_ch = self._selected_audio_device()
        if ref is None:
            return None
        channel = self.audio_channel_var.get()
        is_device = isinstance(ref, int)
        return speech.SpeechSettings(
            engine=self._engine_code(),
            device_index=ref if is_device else None,
            ndi_source=None if is_device else ref,
            channel=None if channel == CHANNEL_MIX_LABEL else int(channel) - 1,
            source_lang=self._source_lang_code(),
            target_lang=self._target_lang_code(),
            whisper_model=self.whisper_model_var.get(),
            hotwords=self.hotwords_var.get(),
            azure_key=self.azure_key_var.get(),
            azure_region=self.azure_region_var.get(),
            soniox_key=self.soniox_key_var.get(),
            speechmatics_key=self.speechmatics_key_var.get(),
            translation=self._translation_code(),
            claude_key=self.claude_key_var.get(),
            claude_model=self.claude_model_var.get(),
        )

    def _update_level_meter(self, level_db):
        frac = min(1.0, max(0.0, (level_db + 60.0) / 60.0))
        color = REC_BG if level_db > -6.0 else PLAY_BG
        for canvas, bar, h in (
            (self.level_canvas, self.level_bar, LEVEL_METER_H),
            (self.card_level_canvas, self.card_level_bar, 6),
        ):
            canvas.coords(bar, 0, 0, frac * LEVEL_METER_W, h + 2)
            canvas.itemconfig(bar, fill=color)
        self.level_text_var.set(f"{level_db:.0f} dB" if level_db > -100 else "— dB")

    def _set_input_settings_locked(self, locked):
        for widget, state in self._lock_while_running:
            widget.config(state="disabled" if locked else state)

    def _on_target_lang_change(self, event=None):
        if hasattr(self, "source_card_var"):
            self._update_source_card()
        if hasattr(self, "voice_lang_combo"):
            self._refresh_voice_langs()

    # ---------- Voice (translated speech output) ----------

    def _open_voice_panel(self):
        self.voice_panel.deiconify()
        self.voice_panel.lift()

    def _build_voice_panel(self):
        win = self._make_panel_window("Voice — translated speech")
        self.voice_panel = win
        prefs = self.audio_prefs

        frame = ttk.Frame(win)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        frame.columnconfigure(1, weight=1)

        def combo(var, values, width, on_change):
            c = ttk.Combobox(frame, textvariable=var, state="readonly", values=values, width=width)
            c.bind("<<ComboboxSelected>>", on_change)
            self._lock_while_running.append((c, "readonly"))
            return c

        ttk.Label(frame, text="Voice:").grid(row=0, column=0, sticky="w", pady=2)
        self.voice_engine_var = tk.StringVar(
            value=VOICE_ENGINE_LABELS.get(prefs.get("voice_engine"), VOICE_ENGINE_LABELS["off"])
        )
        combo(self.voice_engine_var, list(VOICE_ENGINE_LABELS.values()), 24, self._on_voice_engine_change).grid(
            row=0, column=1, sticky="w", padx=4, pady=2
        )

        ttk.Label(frame, text="Voice language:").grid(row=1, column=0, sticky="w", pady=2)
        self.voice_lang_var = tk.StringVar(value=self._lang_label(prefs.get("voice_lang", "same")))
        self.voice_lang_combo = combo(self.voice_lang_var, [], 18, lambda e: self._refresh_voice_names())
        self.voice_lang_combo.grid(row=1, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(frame, text="Speaker:").grid(row=2, column=0, sticky="w", pady=2)
        self.voice_name_var = tk.StringVar(value=prefs.get("voice_name", ""))
        self.voice_name_combo = combo(self.voice_name_var, [], 28, lambda e: self._update_voice_card())
        self.voice_name_combo.grid(row=2, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(frame, text="Output device:").grid(row=3, column=0, sticky="w", pady=2)
        self.voice_device_var = tk.StringVar(value=prefs.get("voice_device_label", NO_DEVICE_LABEL))
        self.voice_device_combo = combo(self.voice_device_var, [NO_DEVICE_LABEL], 42, lambda e: self._update_voice_card())
        self.voice_device_combo.grid(row=3, column=1, sticky="we", padx=4, pady=2)
        self._make_simple_button(
            frame, "Refresh", lambda: self._refresh_audio_devices(rescan=True)
        ).grid(row=3, column=2, padx=4)

        self.voice_ndi_var = tk.BooleanVar(value=bool(prefs.get("voice_ndi", False)))
        ndi_check = ttk.Checkbutton(
            frame, text="Also send it as the audio of the NDI output",
            variable=self.voice_ndi_var, command=self._update_voice_card,
        )
        ndi_check.grid(row=4, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 6))
        self._lock_while_running.append((ndi_check, "normal"))

        if tts is not None:
            self._add_slider(frame, 5, "Speed", 0.7, 1.5, self.voice_tuning.speed, "speed",
                             decimals=2, target=self.voice_tuning)
            self._add_slider(frame, 6, "Volume", 0.0, 2.0, self.voice_tuning.volume, "volume",
                             decimals=2, target=self.voice_tuning)
            self._add_slider(frame, 7, "Audio buffer (ms, on Start)", 50, 1000, self.voice_tuning.buffer_ms,
                             "buffer_ms", target=self.voice_tuning)

        for row, text in enumerate((
            "Speaks whole sentences, a few seconds behind the speaker; it speeds up (up to +35%)"
            " as it falls behind, and skips sentences only past 20 s of delay.",
            "Keep the voice out of the microphone's reach (headphones, interpretation system,"
            " stream) — otherwise it gets translated again.",
            "Azure and Soniox voices use the keys from the Input panel.",
            "If the voice still breaks up, raise the audio buffer (it adds that much delay).",
        ), start=8):
            ttk.Label(frame, text=text, foreground=MUTED_FG, wraplength=520, justify="left").grid(
                row=row, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 0)
            )

        self._refresh_voice_langs()
        self._refresh_output_devices()

    def _voice_engine_code(self):
        return next((k for k, v in VOICE_ENGINE_LABELS.items() if v == self.voice_engine_var.get()), "off")

    def _on_voice_engine_change(self, event=None):
        self._refresh_voice_langs()

    def _voice_lang_choice(self):
        return self._lang_code(self.voice_lang_var.get(), "same")

    def _voice_lang_code(self):
        """The language the voice speaks ("Same as subtitles" resolved)."""
        code = self._voice_lang_choice()
        return self._target_lang_code() if code == "same" else code

    def _refresh_voice_langs(self):
        langs = languages.ordered(languages.voices(self._voice_engine_code()))
        self.voice_lang_combo["values"] = [SAME_LANG_LABEL] + [languages.label(c) for c in langs]
        choice = self._voice_lang_choice()
        if choice != "same" and choice not in langs:
            self.voice_lang_var.set(SAME_LANG_LABEL)
        self._refresh_voice_names()

    def _refresh_voice_names(self):
        names = list(tts.voices_for(self._voice_engine_code(), self._voice_lang_code())) if tts else []
        self.voice_name_combo["values"] = names
        if self.voice_name_var.get() not in names:
            self.voice_name_var.set(names[0] if names else "")
        self._update_voice_card()

    def _refresh_output_devices(self):
        try:
            self.output_devices = audio_io.list_output_devices() if audio_io else []
        except Exception as e:
            self._append_log(f"Could not list audio output devices: {e}")
            self.output_devices = []
        labels = [NO_DEVICE_LABEL] + [label for _idx, label, _ch in self.output_devices]
        self.voice_device_combo["values"] = labels
        if self.voice_device_var.get() not in labels:
            self.voice_device_var.set(NO_DEVICE_LABEL)
        self._update_voice_card()

    def _update_voice_card(self):
        if self.input_mode == "audio" and self._voice_engine_code() != "off":
            self.voice_listen_row.pack(anchor="w", pady=(4, 0))
        else:
            self.voice_listen_row.pack_forget()
        if self.input_mode != "audio":
            self.voice_card_var.set("Audio input mode only")
            return
        engine = self._voice_engine_code()
        if engine == "off":
            self.voice_card_var.set("Off")
            return
        name = self.voice_name_var.get()
        if engine == "azure":
            speaker = f"{name[6:].replace('Neural', '')} (Azure)"
        elif engine == "soniox":
            speaker = f"{name} (Soniox)"
        else:
            speaker = f"{name.split('-')[1] if '-' in name else name} (local)"
        outputs = [o for o, on in (
            ("Device", self.voice_device_var.get() != NO_DEVICE_LABEL), ("NDI", self.voice_ndi_var.get())
        ) if on]
        where = " + ".join(outputs) if outputs else "No output selected"
        self.voice_card_var.set(f"{speaker} · {self._voice_lang_code().upper()}\n{where}")

    def _voice_settings(self):
        engine = self._voice_engine_code()
        if tts is None or engine == "off":
            return None
        label = self.voice_device_var.get()
        device_index = next((idx for idx, lbl, _ch in self.output_devices if lbl == label), None)
        if device_index is None and not self.voice_ndi_var.get() and not self._voice_listening:
            self._append_log(
                "Voice is on but has no output (device or NDI) — it won't be heard. "
                "To only listen to it here, turn on the speaker on the VOICE card before Start."
            )
            return None
        lang = self._voice_lang_code()
        if not self.voice_name_var.get():
            self._append_log(f"No {VOICE_ENGINE_LABELS[engine]} voice speaks {languages.label(lang)} — voice off.")
            return None
        # Offline translation for the voice is only needed (and only exists
        # for EN/PT/ES) when nothing else provides the voice's language: the
        # captions' translation covers it if it's the same language, Azure
        # and Claude translate into a second language too.
        source = self._source_lang_code()
        if lang == self._target_lang_code() or self._engine_code() == "azure" or self._translation_code() == "claude":
            source_langs = ()
        else:
            source_langs = tuple(l for l in (speech.LANGS if source == "auto" else (source,)) if l in speech.LANGS)
        return tts.VoiceSettings(
            engine=engine,
            voice=self.voice_name_var.get(),
            lang=lang,
            source_langs=source_langs,
            device_index=device_index,
            ndi=self.voice_ndi_var.get(),
            azure_key=self.azure_key_var.get(),
            azure_region=self.azure_region_var.get(),
            soniox_key=self.soniox_key_var.get(),
        )

    def _build_text_panel(self):
        win = self._make_panel_window("Text — keying, shadow & color")
        self.text_panel = win

        sliders = ttk.Frame(win)
        sliders.pack(fill="both", expand=True, padx=10, pady=10)
        sliders.columnconfigure(1, weight=1)

        # Keying (NDI video mode) and caption layout (audio mode) share row 0;
        # _show_input_mode_frames() shows the one for the current mode.
        key_settings = ttk.Frame(sliders)
        key_settings.grid(row=0, column=0, columnspan=3, sticky="we")
        key_settings.columnconfigure(1, weight=1)
        self.key_settings_frame = key_settings
        self._build_captions_frame(sliders)

        mode_row = ttk.Frame(key_settings)
        mode_row.grid(row=0, column=0, columnspan=3, sticky="we", padx=4, pady=(0, 6))
        ttk.Label(mode_row, text="Key mode:").pack(side="left")
        self.key_mode_var = tk.StringVar(value=KEY_MODE_LABELS[self.live_cfg.key_mode])
        key_mode_combo = ttk.Combobox(
            mode_row, textvariable=self.key_mode_var, state="readonly",
            values=list(KEY_MODE_LABELS.values()), width=22,
        )
        key_mode_combo.pack(side="left", padx=6)
        key_mode_combo.bind("<<ComboboxSelected>>", self._on_key_mode_change)

        self.luma_frame = ttk.Frame(key_settings)
        self.luma_frame.grid(row=1, column=0, columnspan=3, sticky="we")
        self.luma_frame.columnconfigure(1, weight=1)
        self._add_slider(
            self.luma_frame, 0, "Black level", 0, 255, self.live_cfg.black_level, "black_level",
            on_change=lambda _v: self._update_text_card(),
        )
        self._add_slider(
            self.luma_frame, 1, "White level", 0, 255, self.live_cfg.white_level, "white_level",
            on_change=lambda _v: self._update_text_card(),
        )

        self.color_key_frame = ttk.Frame(key_settings)
        self.color_key_frame.grid(row=1, column=0, columnspan=3, sticky="we")
        self.color_key_frame.columnconfigure(1, weight=1)
        ck_color_row = ttk.Frame(self.color_key_frame)
        ck_color_row.grid(row=0, column=0, columnspan=3, sticky="we", padx=4, pady=(2, 4))
        ttk.Label(ck_color_row, text="Key color:").pack(side="left")
        self.key_color_swatch = self._add_color_picker(
            ck_color_row, "", "key_color_bgr", title="Key color"
        )
        self.pick_btn = self._make_float_button(
            ck_color_row,
            {"idle": ("Pick from preview", "neutral"), "picking": ("Click the preview...", "accent")},
            "idle", command=self._toggle_color_pick,
        )
        self.pick_btn.pack(side="left", padx=(10, 0))
        self._add_slider(
            self.color_key_frame, 1, "Similarity", 0, 441.7, self.live_cfg.key_similarity, "key_similarity"
        )
        self._add_slider(
            self.color_key_frame, 2, "Smoothness", 1, 200, self.live_cfg.key_smoothness, "key_smoothness"
        )
        if self.live_cfg.key_mode == "color":
            self.luma_frame.grid_remove()
        else:
            self.color_key_frame.grid_remove()

        self._add_slider(sliders, 2, "Shadow X (px)", -20, 20, self.live_cfg.shadow_dx, "shadow_dx")
        self._add_slider(sliders, 3, "Shadow Y (px)", -20, 20, self.live_cfg.shadow_dy, "shadow_dy")
        self._add_slider(sliders, 4, "Shadow blur", 0, 20, self.live_cfg.shadow_blur_sigma, "shadow_blur_sigma")
        self._add_slider(
            sliders, 5, "Shadow opacity", 0, 1, self.live_cfg.shadow_opacity, "shadow_opacity", decimals=2
        )

        colors_row = ttk.Frame(sliders)
        colors_row.grid(row=6, column=0, columnspan=3, sticky="we", padx=4, pady=(4, 0))
        self.text_color_swatch = self._add_color_picker(colors_row, "Text color:", "text_color_bgr")
        ttk.Label(colors_row, text="   Shadow color:").pack(side="left")
        self.color_swatch = self._add_color_picker(
            colors_row, "", "shadow_color_bgr", title="Shadow color"
        )
        self._show_input_mode_frames()

    def _build_captions_frame(self, parent):
        """Caption layout for the audio mode: font, number of roll-up lines,
        line width, position and how long captions stay up after speech."""
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=0, columnspan=3, sticky="we", pady=(0, 6))
        frame.columnconfigure(1, weight=1)
        self.captions_frame = frame
        cap = self.caption_cfg
        refresh = lambda _v: self._update_text_card()  # noqa: E731

        self._add_slider(frame, 0, "Font size (px)", 16, 160, cap.font_size, "font_size",
                         target=cap, on_change=refresh)
        self._add_slider(frame, 1, "Lines", 1, 4, cap.max_lines, "max_lines", target=cap, on_change=refresh)
        self._add_slider(frame, 2, "Line width (% of output)", 20, 100, cap.line_width_pct,
                         "line_width_pct", target=cap)
        self._add_slider(frame, 3, "Bottom margin (px)", 0, 1000, cap.bottom_margin, "bottom_margin", target=cap)
        self._add_slider(frame, 4, "Clear after silence (s)", 1, 15, cap.hold_s, "hold_s", decimals=1, target=cap)
        self.caption_bold_var = tk.BooleanVar(value=cap.bold)
        ttk.Checkbutton(
            frame, text="Bold", variable=self.caption_bold_var,
            command=lambda: setattr(self.caption_cfg, "bold", self.caption_bold_var.get()),
        ).grid(row=5, column=0, sticky="w", padx=4, pady=(2, 0))
        self.caption_live_var = tk.BooleanVar(value=cap.live)
        ttk.Checkbutton(
            frame, text="Show text while speaking (faster, but it corrects itself on screen)",
            variable=self.caption_live_var,
            command=lambda: setattr(self.caption_cfg, "live", self.caption_live_var.get()),
        ).grid(row=6, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 0))

    def _build_background_panel(self):
        win = self._make_panel_window("Background box")
        self.background_panel = win

        frame = ttk.Frame(win)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        frame.columnconfigure(1, weight=1)

        top_row = ttk.Frame(frame)
        top_row.grid(row=0, column=0, columnspan=3, sticky="we", padx=4, pady=(0, 4))
        ttk.Checkbutton(
            top_row, text="Enabled", variable=self.box_enabled_var,
            command=self._toggle_box_enabled,
        ).pack(side="left")
        ttk.Label(top_row, text="   Box color:").pack(side="left")
        self.box_color_swatch = self._add_color_picker(
            top_row, "", "box_color_bgr", title="Background box color"
        )

        self._add_slider(
            frame, 1, "Box opacity", 0, 1, self.live_cfg.box_opacity, "box_opacity", decimals=2,
            on_change=lambda _v: self._update_background_card(),
        )
        self._add_slider(
            frame, 2, "Box width (px, 0 = full)", 0, BOX_SIZE_MAX, self.live_cfg.box_w, "box_w"
        )
        self._add_slider(
            frame, 3, "Box height (px, 0 = full)", 0, BOX_SIZE_MAX, self.live_cfg.box_h, "box_h"
        )
        self._add_slider(
            frame, 4, "Box position X (px, + right)", -BOX_POS_RANGE, BOX_POS_RANGE,
            self.live_cfg.box_pos_x, "box_pos_x",
        )
        self._add_slider(
            frame, 5, "Box position Y (px, + up)", -BOX_POS_RANGE, BOX_POS_RANGE,
            self.live_cfg.box_pos_y, "box_pos_y",
        )

    def _build_presets_ui(self, pad):
        presets_frame = ttk.LabelFrame(
            self.content, text="PRESETS   ·   tap = apply   ·   hold = save"
        )
        presets_frame.pack(fill="x", padx=pad["padx"], pady=2)
        for i in range(1, PRESET_SLOTS + 1):
            slot = str(i)
            col = ttk.Frame(presets_frame)
            col.pack(side="left", padx=6, pady=3)
            self._make_preset_button(col, slot).pack(side="left")

    def _make_preset_button(self, parent, slot):
        w, h, radius = 30, 26, BUTTON_RADIUS
        idle_img = ImageTk.PhotoImage(
            self._flat_rect(w, h, radius, self._hex_to_rgba(BG), self._hex_to_rgba(BORDER))
        )
        idle_hover_img = ImageTk.PhotoImage(
            self._flat_rect(w, h, radius, self._hex_to_rgba(PANEL_BG), self._hex_to_rgba(BORDER))
        )
        launch_img = ImageTk.PhotoImage(
            self._flat_rect(w, h, radius, self._hex_to_rgba(PLAY_BG), self._hex_to_rgba(App._shade(PLAY_BG, 0.7)))
        )
        rec_img = ImageTk.PhotoImage(
            self._flat_rect(w, h, radius, self._hex_to_rgba(REC_BG), self._hex_to_rgba(App._shade(REC_BG, 0.7)))
        )

        btn = tk.Label(
            parent, text=slot, image=idle_img, compound="center", fg=FG,
            font=self.normal_font, cursor="hand2", bd=0, highlightthickness=0, bg=BG,
        )
        btn._images = (idle_img, idle_hover_img, launch_img, rec_img)  # keep a live reference
        state = {"timer": None, "long_fired": False}

        def show_idle(hover=False):
            btn.config(text=slot, image=idle_hover_img if hover else idle_img, fg=FG, font=self.normal_font)

        def cancel_timer():
            if state["timer"] is not None:
                btn.after_cancel(state["timer"])
                state["timer"] = None

        def fire_long_press():
            state["timer"] = None
            state["long_fired"] = True
            self._save_preset(slot)
            # Visual confirmation that it was saved: flashes red, then returns to the number.
            btn.config(text="●", image=rec_img, fg="white", font=self.bold_font)
            btn.after(SAVE_FLASH_MS, show_idle)

        def on_press(event):
            state["long_fired"] = False
            # Pressing shows the "about to launch" state: green background + play arrow.
            btn.config(text="▶", image=launch_img, fg="white", font=self.bold_font)
            state["timer"] = btn.after(LONG_PRESS_MS, fire_long_press)

        def on_release(event):
            cancel_timer()
            if state["long_fired"]:
                return  # already handled (save) by the long-press
            inside = 0 <= event.x < btn.winfo_width() and 0 <= event.y < btn.winfo_height()
            show_idle(hover=inside)
            if inside:
                self._apply_preset(slot)

        def on_enter(event):
            if not state["long_fired"]:
                show_idle(hover=True)

        def on_leave(event):
            cancel_timer()
            if not state["long_fired"]:
                show_idle(hover=False)

        btn.bind("<ButtonPress-1>", on_press)
        btn.bind("<ButtonRelease-1>", on_release)
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn

    @staticmethod
    def _darken_titlebar(win):
        """On macOS, requests a dark title bar to match the rest of the theme
        (no effect on other platforms)."""
        try:
            win.update_idletasks()
            win.tk.call("::tk::unsupported::MacWindowStyle", "appearance", win, "dark")
        except tk.TclError:
            pass

    @staticmethod
    def _shade(hex_color, factor):
        """Darkens (factor<1) or lightens (factor>1) a #rrggbb color."""
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        r, g, b = (max(0, min(255, int(c * factor))) for c in (r, g, b))
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def _hex_to_rgba(hex_color, alpha=255):
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return (r, g, b, alpha)

    @staticmethod
    def _flat_rect(w, h, radius, fill_rgba, border_rgba):
        """Draws a near-flat rectangle (small radius, no drop shadow) with a
        thin 1px border — broadcast-console look. `fill_rgba` may be None for
        an outline-only button. ttk has no border-radius support, so we render
        with PIL instead."""
        pad = 2
        W, H = w + pad * 2, h + pad * 2
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        box = [pad, pad, pad + w, pad + h]
        if fill_rgba is not None:
            draw.rounded_rectangle(box, radius=radius, fill=fill_rgba)
        draw.rounded_rectangle(box, radius=radius, outline=border_rgba, width=1)
        return img

    def _make_float_button(self, parent, variants, initial, command=None, height=26, radius=BUTTON_RADIUS, min_width=0):
        """Flat button with a thin border (broadcast-console look), rendered
        with PIL because ttk's 'clam' theme has no border-radius support.
        `variants` is {name: (text, kind)} with kind from BUTTON_KINDS; all
        variants share the width of the widest one, so the button doesn't
        "jump" when its text changes (e.g. Start/Stop). Call `set_variant(name)`
        afterwards to switch. A variant may carry a third item, an icon
        function (w, h, rgba) -> RGBA image drawn over the button, in place
        of or next to its text."""
        measure_font = tkfont.Font(family=self.font_family, size=10, weight="bold")
        text_w = max(measure_font.measure(v[0]) for v in variants.values())
        w = max(text_w + 26, min_width)

        image_sets = {}
        for name, (text, kind, *icon) in variants.items():
            fill_hex, fg, bold = BUTTON_KINDS[kind]
            if fill_hex is None:
                idle_hex, hover_hex, press_hex, border_hex = BG, PANEL_BG, FIELD_BG, BORDER
            else:
                idle_hex = fill_hex
                hover_hex = App._shade(fill_hex, 1.15)
                press_hex = App._shade(fill_hex, 0.75)
                border_hex = App._shade(fill_hex, 0.7)

            def image(fill_hex, border_hex=border_hex, fg=fg, icon=icon):
                img = self._flat_rect(w, height, radius, self._hex_to_rgba(fill_hex), self._hex_to_rgba(border_hex))
                if icon:
                    img = Image.alpha_composite(img, icon[0](img.width, img.height, self._hex_to_rgba(fg)))
                return ImageTk.PhotoImage(img)

            image_sets[name] = {
                "text": text,
                "fg": fg,
                "font": self.bold_font if bold else self.normal_font,
                "idle": image(idle_hex),
                "hover": image(hover_hex),
                "press": image(press_hex),
            }

        btn = tk.Label(
            parent, compound="center", bg=BG,
            cursor="hand2", bd=0, highlightthickness=0,
        )
        btn._image_sets = image_sets  # keep a live reference (avoids garbage collection)
        btn._variant = initial
        btn._pressed = False

        def apply(state):
            data = image_sets[btn._variant]
            btn.config(image=data[state], text=data["text"], fg=data["fg"], font=data["font"])

        def set_variant(name):
            btn._variant = name
            apply("press" if btn._pressed else "idle")

        def on_enter(event):
            if not btn._pressed:
                apply("hover")

        def on_leave(event):
            if not btn._pressed:
                apply("idle")

        def on_press(event):
            btn._pressed = True
            apply("press")

        def on_release(event):
            btn._pressed = False
            inside = 0 <= event.x < btn.winfo_width() and 0 <= event.y < btn.winfo_height()
            apply("hover" if inside else "idle")
            if inside and command:
                command()

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        btn.bind("<ButtonPress-1>", on_press)
        btn.bind("<ButtonRelease-1>", on_release)

        btn.set_variant = set_variant
        set_variant(initial)
        return btn

    @staticmethod
    def _speaker_icon(waves):
        """Icon function for _make_float_button: a loudspeaker with sound
        waves (`waves`) or crossed out. Drawn rather than an emoji, which
        Tk renders unreliably (especially on Windows)."""

        def draw(w, h, rgba):
            s = 4  # supersampling, for smooth edges once scaled down
            img = Image.new("RGBA", (w * s, h * s), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            u = min(w, h) * 0.065 * s
            cx, cy = w * s / 2, h * s / 2
            left = cx - 5 * u
            d.polygon(
                [(left, cy - 1.6 * u), (left + 2.4 * u, cy - 1.6 * u), (left + 5 * u, cy - 4.2 * u),
                 (left + 5 * u, cy + 4.2 * u), (left + 2.4 * u, cy + 1.6 * u), (left, cy + 1.6 * u)],
                fill=rgba,
            )
            stroke = max(s, int(0.9 * u))
            if waves:
                ax = left + 5 * u
                for r in (2.4 * u, 4.6 * u):
                    d.arc([ax - r, cy - r, ax + r, cy + r], start=-50, end=50, fill=rgba, width=stroke)
            else:
                x0, x1 = left + 6.6 * u, left + 9.8 * u
                d.line([(x0, cy - 1.6 * u), (x1, cy + 1.6 * u)], fill=rgba, width=stroke)
                d.line([(x0, cy + 1.6 * u), (x1, cy - 1.6 * u)], fill=rgba, width=stroke)
            return img.resize((w, h), Image.LANCZOS)

        return draw

    def _make_listen_button(self, parent, height, bg=BG, voice=False):
        """Speaker toggle that plays, on the computer's default output, the
        selected audio input (see _toggle_listen) or, with `voice`, the
        translated voice (see _toggle_voice_listen). Every copy of a toggle
        (e.g. input card and input panel) follows the same state."""
        on = self._voice_listening if voice else self._listening
        btn = self._make_float_button(
            parent,
            {"off": ("", "neutral", self._speaker_icon(False)), "on": ("", "accent", self._speaker_icon(True))},
            "on" if on else "off", command=self._toggle_voice_listen if voice else self._toggle_listen,
            height=height, min_width=height + 8,
        )
        btn.config(bg=bg)
        (self._voice_listen_btns if voice else self._listen_btns).append(btn)
        return btn

    def _make_simple_button(self, parent, text, command, kind="neutral", height=26, min_width=0):
        return self._make_float_button(
            parent, {"default": (text, kind)}, "default", command, height=height, min_width=min_width
        )

    def _build_output_panel(self):
        win = self._make_panel_window("Output — resolution, fps and signal position")
        self.output_panel = win

        frame = ttk.Frame(win)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        name_row = ttk.Frame(frame)
        name_row.pack(fill="x", pady=(0, 8))
        ttk.Label(name_row, text="Output name:").pack(side="left")
        self.output_name_var = tk.StringVar(value=self.audio_prefs.get("output_name") or APP_NAME)
        self.output_name_var.trace_add("write", lambda *a: self._update_output_card())
        ttk.Entry(name_row, textvariable=self.output_name_var).pack(
            side="left", fill="x", expand=True, padx=6
        )

        res_row = ttk.Frame(frame)
        res_row.pack(fill="x", padx=4, pady=(4, 2))
        ttk.Label(res_row, text="Resolution:").pack(side="left")
        self.resolution_var = tk.StringVar(
            value=self._match_resolution_label(self.output_cfg.canvas_w, self.output_cfg.canvas_h)
        )
        res_combo = ttk.Combobox(
            res_row, textvariable=self.resolution_var, state="readonly",
            values=list(RESOLUTION_PRESETS.keys()), width=24,
        )
        res_combo.pack(side="left", padx=(4, 10))
        res_combo.bind("<<ComboboxSelected>>", self._on_resolution_preset)

        spin_kwargs = dict(
            bg=FIELD_BG, fg=FG, buttonbackground=PANEL_BG, insertbackground=FG,
            relief="flat", highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=ACCENT, disabledbackground=FIELD_BG,
        )
        self.out_width_var = tk.IntVar(value=self.output_cfg.canvas_w)
        self.out_height_var = tk.IntVar(value=self.output_cfg.canvas_h)
        w_spin = tk.Spinbox(
            res_row, from_=16, to=7680, increment=2, width=6,
            textvariable=self.out_width_var, command=self._apply_output_resolution, **spin_kwargs,
        )
        w_spin.pack(side="left")
        ttk.Label(res_row, text="x").pack(side="left", padx=3)
        h_spin = tk.Spinbox(
            res_row, from_=16, to=7680, increment=2, width=6,
            textvariable=self.out_height_var, command=self._apply_output_resolution, **spin_kwargs,
        )
        h_spin.pack(side="left")
        for spin in (w_spin, h_spin):
            spin.bind("<Return>", self._apply_output_resolution)
            spin.bind("<FocusOut>", self._apply_output_resolution)

        ttk.Label(res_row, text="   FPS:").pack(side="left", padx=(14, 4))
        self.fps_var = tk.StringVar(
            value=self._match_fps_label(self.output_cfg.fps_n, self.output_cfg.fps_d)
        )
        fps_combo = ttk.Combobox(
            res_row, textvariable=self.fps_var, state="readonly",
            values=list(FPS_PRESETS.keys()), width=7,
        )
        fps_combo.pack(side="left")
        fps_combo.bind("<<ComboboxSelected>>", self._on_fps_preset)

        ttk.Label(res_row, textvariable=self.output_info_var, foreground=MUTED_FG).pack(
            side="left", padx=(14, 0)
        )

        transform = ttk.Frame(frame)
        transform.pack(fill="x", padx=4, pady=(2, 4))
        transform.columnconfigure(1, weight=1)

        self._add_slider(
            transform, 0, "Signal scale", 0.1, 3.0, self.output_cfg.scale,
            "scale", decimals=2, target=self.output_cfg,
        )
        self._add_slider(
            transform, 1, "Position X (px, + right)", -POS_RANGE, POS_RANGE, self.output_cfg.pos_x_px,
            "pos_x_px", decimals=0, target=self.output_cfg,
        )
        self._add_slider(
            transform, 2, "Position Y (px, + up)", -POS_RANGE, POS_RANGE, self.output_cfg.pos_y_px,
            "pos_y_px", decimals=0, target=self.output_cfg,
        )
        self._make_simple_button(transform, "Center", self._center_transform).grid(
            row=3, column=0, columnspan=3, pady=(4, 0)
        )

    def _match_resolution_label(self, w, h):
        for label, dims in RESOLUTION_PRESETS.items():
            if dims == (w, h):
                return label
        return "Custom"

    def _match_fps_label(self, n, d):
        for label, nd in FPS_PRESETS.items():
            if nd == (n, d):
                return label
        return "29.97"

    def _on_resolution_preset(self, event=None):
        dims = RESOLUTION_PRESETS.get(self.resolution_var.get())
        if dims:
            self.out_width_var.set(dims[0])
            self.out_height_var.set(dims[1])
            self._apply_output_resolution()

    def _apply_output_resolution(self, event=None):
        try:
            w = int(self.out_width_var.get())
            h = int(self.out_height_var.get())
        except (tk.TclError, ValueError):
            return
        w = max(16, min(7680, w))
        h = max(16, min(7680, h))
        self.output_cfg.canvas_w = w
        self.output_cfg.canvas_h = h
        self._update_output_info_label()
        self._update_output_card()

    def _on_fps_preset(self, event=None):
        nd = FPS_PRESETS.get(self.fps_var.get())
        if nd:
            self.output_cfg.fps_n, self.output_cfg.fps_d = nd
            self._update_output_info_label()
            self._update_output_card()

    def _center_transform(self):
        self.sliders["scale"].set(1.0)
        self.sliders["pos_x_px"].set(0.0)
        self.sliders["pos_y_px"].set(0.0)

    @staticmethod
    def _fmt_fps(n, d):
        if not d:
            return "?"
        s = f"{n / d:.2f}".rstrip("0").rstrip(".")
        return s

    def _update_output_info_label(self):
        fps_s = self._fmt_fps(self.output_cfg.fps_n, self.output_cfg.fps_d)
        self.output_info_var.set(
            f"Output: {self.output_cfg.canvas_w}x{self.output_cfg.canvas_h} @ {fps_s} fps"
        )

    def _open_preview_window(self):
        if self.preview_window is not None and self.preview_window.winfo_exists():
            self.preview_window.deiconify()
            self.preview_window.lift()
            return
        self._build_preview_window()

    def _close_preview_window(self):
        if self.picking_color:
            self._end_color_pick()
        if self.preview_window is not None:
            self.preview_window.destroy()
        self.preview_window = None
        self.preview_canvas = None

    def _build_preview_window(self):
        win = tk.Toplevel(self.root)
        win.title("Preview (composited over gray, to check alpha)")
        win.geometry("560x480")
        win.minsize(360, 280)
        win.configure(bg=BG)
        self._darken_titlebar(win)
        win.protocol("WM_DELETE_WINDOW", self._close_preview_window)
        self.preview_window = win

        zoom_row = ttk.Frame(win)
        zoom_row.pack(fill="x", padx=4, pady=(4, 0))
        ttk.Label(zoom_row, text="Zoom:").pack(side="left")
        self._make_simple_button(
            zoom_row, "-", lambda: self._nudge_zoom(-0.25), min_width=28
        ).pack(side="left", padx=(4, 0))
        self.zoom_label = ttk.Label(zoom_row, text="1.00x", width=6)
        self.zoom_scale = ttk.Scale(
            zoom_row, from_=ZOOM_MIN, to=ZOOM_MAX, orient="horizontal", command=self._on_zoom_change
        )
        self.zoom_scale.pack(side="left", fill="x", expand=True, padx=4)
        self._make_simple_button(
            zoom_row, "+", lambda: self._nudge_zoom(0.25), min_width=28
        ).pack(side="left")
        self.zoom_label.pack(side="left", padx=(4, 0))
        self.zoom_scale.set(self.zoom)
        ttk.Label(zoom_row, text="(drag the image to pan when zoomed in)").pack(
            side="left", padx=(10, 0)
        )

        canvas_holder = ttk.Frame(win)
        canvas_holder.pack(fill="both", expand=True, padx=4, pady=4)
        canvas_holder.rowconfigure(0, weight=1)
        canvas_holder.columnconfigure(0, weight=1)

        self.preview_canvas = tk.Canvas(canvas_holder, bg="#3a3a3a")
        hbar = ttk.Scrollbar(canvas_holder, orient="horizontal", command=self.preview_canvas.xview)
        vbar = ttk.Scrollbar(canvas_holder, orient="vertical", command=self.preview_canvas.yview)
        self.preview_canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="we")

        self.preview_canvas.bind("<ButtonPress-1>", self._on_preview_press)
        self.preview_canvas.bind("<B1-Motion>", self._on_preview_drag)

    def _on_preview_press(self, event):
        if self.picking_color:
            self._on_pick_click(event)
        else:
            self.preview_canvas.scan_mark(event.x, event.y)

    def _on_preview_drag(self, event):
        if not self.picking_color:
            self.preview_canvas.scan_dragto(event.x, event.y, gain=1)

    def _add_slider(self, parent, row, label, lo, hi, value, attr, decimals=0, target=None, on_change=None):
        target = self.live_cfg if target is None else target
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        val_var = tk.StringVar(value=self._fmt(value, decimals))

        def _apply(v):
            fv = float(v)
            setattr(target, attr, fv)
            val_var.set(self._fmt(fv, decimals))
            if on_change:
                on_change(fv)

        scale = ttk.Scale(parent, from_=lo, to=hi, orient="horizontal", command=_apply)
        scale.grid(row=row, column=1, sticky="we", padx=4, pady=2)

        def commit_entry(event=None):
            try:
                fv = float(val_var.get().replace(",", "."))
            except ValueError:
                val_var.set(self._fmt(scale.get(), decimals))
                return
            scale.set(min(hi, max(lo, fv)))

        val_entry = ttk.Entry(parent, textvariable=val_var, width=7, justify="right")
        val_entry.grid(row=row, column=2, sticky="e", padx=4)
        val_entry.bind("<Return>", commit_entry)
        val_entry.bind("<FocusOut>", commit_entry)

        scale.set(value)
        self.sliders[attr] = scale

    @staticmethod
    def _fmt(v, decimals):
        return f"{v:.{decimals}f}"

    @staticmethod
    def _bgr_to_hex(bgr):
        b, g, r = (int(c) for c in bgr)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _add_color_picker(self, parent, label, attr, title=None):
        """Builds a "label + swatch + Choose..." color picker bound to a
        live_cfg BGR-tuple attribute, and returns the swatch Canvas (so callers
        can update it later, e.g. when applying a preset)."""
        if label:
            ttk.Label(parent, text=label).pack(side="left")
        swatch = tk.Canvas(
            parent, width=24, height=18,
            bg=self._bgr_to_hex(getattr(self.live_cfg, attr)),
            highlightthickness=1, highlightbackground=BORDER,
        )
        swatch.pack(side="left", padx=6)
        dialog_title = title or (label.rstrip(":").strip() if label else attr)
        self._make_simple_button(
            parent, "Choose...", lambda: self._pick_color(attr, swatch, dialog_title)
        ).pack(side="left")
        return swatch

    def _pick_color(self, attr, swatch, title):
        _, hexcolor = colorchooser.askcolor(
            color=self._bgr_to_hex(getattr(self.live_cfg, attr)), title=title
        )
        if hexcolor:
            r = int(hexcolor[1:3], 16)
            g = int(hexcolor[3:5], 16)
            b = int(hexcolor[5:7], 16)
            setattr(self.live_cfg, attr, (b, g, r))
            swatch.config(bg=hexcolor)
            if attr == "box_color_bgr":
                self._update_background_card()
            else:
                self._update_text_card()

    def _toggle_box_enabled(self):
        self.live_cfg.box_enabled = self.box_enabled_var.get()
        self._update_background_card()

    # ---------- Key mode ----------

    def _on_key_mode_change(self, event=None):
        mode = "color" if self.key_mode_var.get() == KEY_MODE_LABELS["color"] else "luma"
        self._set_key_mode(mode)

    def _set_key_mode(self, mode):
        self.live_cfg.key_mode = mode
        self.key_mode_var.set(KEY_MODE_LABELS[mode])
        if mode == "color":
            self.luma_frame.grid_remove()
            self.color_key_frame.grid()
        else:
            if self.picking_color:
                self._end_color_pick()
            self.color_key_frame.grid_remove()
            self.luma_frame.grid()
        self._update_text_card()

    # ---------- Color pick (eyedropper) ----------

    def _toggle_color_pick(self):
        if self.picking_color:
            self._end_color_pick()
            return
        if self.last_raw_bgr is None:
            messagebox.showinfo(
                APP_NAME, "Start the pipeline first so there is a live preview to pick from."
            )
            return
        self._open_preview_window()
        self.picking_color = True
        self.pick_btn.set_variant("picking")
        self._append_log("Click a pixel in the preview to set the key color (Esc to cancel).")
        self._render_picking_preview()
        self.preview_canvas.config(cursor="crosshair")
        self.root.bind("<Escape>", self._end_color_pick)

    def _end_color_pick(self, event=None):
        self.picking_color = False
        self.pick_btn.set_variant("idle")
        if self.preview_canvas is not None and self.preview_canvas.winfo_exists():
            self.preview_canvas.config(cursor="")
        self.root.unbind("<Escape>")

    def _render_picking_preview(self):
        if self.last_raw_bgr is None:
            return
        if self.preview_canvas is None or not self.preview_canvas.winfo_exists():
            return
        bgr = self.last_raw_bgr
        h, w = bgr.shape[:2]
        base_scale = min(1.0, PREVIEW_MAX_W / w)
        total_scale = base_scale * self.zoom
        if total_scale != 1.0:
            new_w = max(1, int(w * total_scale))
            new_h = max(1, int(h * total_scale))
            interp = cv2.INTER_AREA if total_scale < 1.0 else cv2.INTER_LINEAR
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=interp)
        self._pick_scale = total_scale

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb, mode="RGB")
        self.preview_photo = ImageTk.PhotoImage(img)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, image=self.preview_photo, anchor="nw")
        self.preview_canvas.config(scrollregion=(0, 0, bgr.shape[1], bgr.shape[0]))
        self.preview_canvas.create_text(
            12, 10, text="Click to pick the key color  ·  Esc to cancel", anchor="nw",
            fill=ACCENT, font=(self.font_family, 12, "bold"),
        )

    def _on_pick_click(self, event):
        if self.last_raw_bgr is None:
            return
        cx = self.preview_canvas.canvasx(event.x)
        cy = self.preview_canvas.canvasy(event.y)
        scale = self._pick_scale or 1.0
        raw = self.last_raw_bgr
        h, w = raw.shape[:2]
        x = max(0, min(w - 1, int(cx / scale)))
        y = max(0, min(h - 1, int(cy / scale)))
        b, g, r = (int(c) for c in raw[y, x])
        self.live_cfg.key_color_bgr = (b, g, r)
        self.key_color_swatch.config(bg=self._bgr_to_hex((b, g, r)))
        self._append_log(f"Key color picked: RGB({r}, {g}, {b}).")
        self._update_text_card()
        self._end_color_pick()

    # ---------- Zoom ----------

    def _on_zoom_change(self, v):
        self.zoom = float(v)
        self.zoom_label.config(text=f"{self.zoom:.2f}x")

    def _nudge_zoom(self, delta):
        new_val = min(ZOOM_MAX, max(ZOOM_MIN, self.zoom + delta))
        self.zoom_scale.set(new_val)

    # ---------- Presets ----------

    def _load_presets(self):
        try:
            with open(PRESETS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        return {}

    def _save_presets_to_disk(self):
        try:
            with open(PRESETS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.presets, f, indent=2)
        except OSError as e:
            self._append_log(f"Could not save presets: {e}")

    def _load_output_defaults(self):
        cfg = OutputConfig()
        data = self.presets.get("_output")
        if isinstance(data, dict):
            for attr in OUTPUT_ATTRS:
                if attr in data:
                    setattr(cfg, attr, data[attr])
        return cfg

    def _save_output_defaults(self):
        cfg = self.output_cfg
        self.presets["_output"] = {attr: getattr(cfg, attr) for attr in OUTPUT_ATTRS}
        self._save_presets_to_disk()

    def _load_audio_defaults(self):
        """Caption layout + audio input settings from the last session. The
        Azure key is kept in the same local, per-user presets file."""
        cfg = CaptionConfig()
        data = self.presets.get("_audio")
        prefs = data if isinstance(data, dict) else {}
        for attr in CAPTION_ATTRS:
            if attr in prefs:
                setattr(cfg, attr, float(prefs[attr]))
        cfg.bold = bool(prefs.get("caption_bold", cfg.bold))
        cfg.live = bool(prefs.get("caption_live", cfg.live))
        return cfg, prefs

    def _save_audio_defaults(self):
        """The last session: the whole app configuration + the API keys and
        Azure region (app-wide) + the caption layout."""
        data = self._collect_app_settings()
        data["azure_key"] = self.azure_key_var.get()
        data["azure_region"] = self.azure_region_var.get()
        for service in ("soniox", "speechmatics", "claude"):
            data[f"{service}_key"] = getattr(self, f"{service}_key_var").get()
        data.update({attr: getattr(self.caption_cfg, attr) for attr in CAPTION_ATTRS})
        data["caption_bold"] = self.caption_cfg.bold
        data["caption_live"] = self.caption_cfg.live
        self.presets["_audio"] = data
        self._save_presets_to_disk()

    def _collect_app_settings(self):
        """Everything but the look (see APP_SETTING_KEYS)."""
        values = {
            "input_mode": self.input_mode,
            "ndi_source_name": self._current_ndi_source_name(),
            "output_name": self.output_name_var.get().strip() or APP_NAME,
            "device_label": self.audio_device_var.get(),
            "channel": self.audio_channel_var.get(),
            "source_lang": self._source_lang_code(),
            "target_lang": self._target_lang_code(),
            "engine": self._engine_code(),
            "whisper_model": self.whisper_model_var.get(),
            "hotwords": self.hotwords_var.get(),
            "translation": self._translation_code(),
            "claude_model": self.claude_model_var.get(),
            "voice_engine": self._voice_engine_code(),
            "voice_name": self.voice_name_var.get(),
            "voice_lang": self._voice_lang_choice(),
            "voice_device_label": self.voice_device_var.get(),
            "voice_ndi": self.voice_ndi_var.get(),
        }
        if self.voice_tuning is not None:
            values["voice_speed"] = self.voice_tuning.speed
            values["voice_volume"] = self.voice_tuning.volume
            values["voice_buffer_ms"] = self.voice_tuning.buffer_ms
        return {key: values[key] for key in APP_SETTING_KEYS if key in values}

    def _apply_app_settings(self, values):
        """Applies a _collect_app_settings() dict to the UI (only while
        stopped: these settings are read on Start). Missing keys are left as
        they are; a device, NDI source or voice that isn't available here is
        kept as the wanted choice (NDI sources may show up later)."""
        if "output_name" in values:
            self.output_name_var.set(values["output_name"] or APP_NAME)
        if values.get("ndi_source_name"):
            self._select_ndi_source(values["ndi_source_name"])
        if speech is not None:
            choices = (
                ("engine", self.engine_var, ENGINE_LABELS),
                ("translation", self.translation_var, TRANSLATION_LABELS),
                ("voice_engine", self.voice_engine_var, VOICE_ENGINE_LABELS),
            )
            for key, var, labels in choices:
                if values.get(key) in labels:
                    var.set(labels[values[key]])
            # Languages the engine doesn't offer fall back in _refresh_lang_choices.
            for key, var in (
                ("source_lang", self.source_lang_var), ("target_lang", self.target_lang_var),
                ("voice_lang", self.voice_lang_var),
            ):
                code = values.get(key)
                if code in languages.NAMES or code in ("auto", "same"):
                    var.set(self._lang_label(code))
            self._show_engine_frames()
            self._refresh_lang_choices()
            for key, var in (("whisper_model", self.whisper_model_var), ("hotwords", self.hotwords_var)):
                if key in values:
                    var.set(values[key])
            if values.get("claude_model") in speech.CLAUDE_MODELS:
                self.claude_model_var.set(values["claude_model"])
            if "device_label" in values:
                self.audio_device_var.set(values["device_label"])
                self._on_audio_device_change()
            if values.get("channel") in self.audio_channel_combo["values"]:
                self.audio_channel_var.set(values["channel"])
            self._refresh_voice_names()
            if values.get("voice_name") in self.voice_name_combo["values"]:
                self.voice_name_var.set(values["voice_name"])
            if values.get("voice_device_label") in self.voice_device_combo["values"]:
                self.voice_device_var.set(values["voice_device_label"])
            if "voice_ndi" in values:
                self.voice_ndi_var.set(bool(values["voice_ndi"]))
            for key, attr in (("voice_speed", "speed"), ("voice_volume", "volume"), ("voice_buffer_ms", "buffer_ms")):
                if key in values and attr in self.sliders:
                    self.sliders[attr].set(float(values[key]))
        mode = values.get("input_mode")
        if mode in INPUT_MODE_LABELS and not (mode == "audio" and speech is None):
            self.input_mode_var.set(INPUT_MODE_LABELS[mode])
            self._on_input_mode_change()
        else:
            self._restart_monitor()
        self._update_source_card()
        self._update_voice_card()
        self._update_output_card()

    def _current_ndi_source_name(self):
        idx = self.source_combo.current()
        if 0 <= idx < len(self.sources):
            return self.sources[idx].ndi_name
        return self._wanted_ndi_source or ""

    def _select_ndi_source(self, name):
        """Selects the NDI (key mode) source `name` now if it's on the
        network, or as soon as discovery finds it."""
        names = [s.ndi_name for s in self.sources]
        if name in names:
            self.source_combo.current(names.index(name))
            self._wanted_ndi_source = None
        else:
            self._wanted_ndi_source = name
            self.source_var.set(name)
        self._update_source_card()

    def _save_preset(self, slot):
        cfg = self.live_cfg
        self.presets[slot] = {attr: getattr(cfg, attr) for attr in PRESET_ATTRS}
        self.presets[slot].update({attr: list(getattr(cfg, attr)) for attr in PRESET_COLOR_ATTRS})
        self.presets[slot]["box_enabled"] = cfg.box_enabled
        self.presets[slot]["key_mode"] = cfg.key_mode
        self.presets[slot].update({attr: getattr(self.output_cfg, attr) for attr in OUTPUT_ATTRS})
        self.presets[slot].update({attr: getattr(self.caption_cfg, attr) for attr in CAPTION_ATTRS})
        self.presets[slot]["caption_bold"] = self.caption_cfg.bold
        self.presets[slot]["caption_live"] = self.caption_cfg.live
        self.presets[slot]["_app"] = self._collect_app_settings()
        self._save_presets_to_disk()
        self._append_log(f"Preset {slot} saved (look + input, recognition, voice and output name).")

    def _apply_preset(self, slot):
        preset = self.presets.get(slot)
        if not preset:
            self._append_log(f"Preset {slot} is empty.")
            return
        for attr in PRESET_ATTRS + CAPTION_ATTRS:
            if attr in preset and attr in self.sliders:
                self.sliders[attr].set(preset[attr])
        if "caption_bold" in preset:
            self.caption_cfg.bold = bool(preset["caption_bold"])
            self.caption_bold_var.set(self.caption_cfg.bold)
        if "caption_live" in preset:
            self.caption_cfg.live = bool(preset["caption_live"])
            self.caption_live_var.set(self.caption_cfg.live)
        color_swatches = {
            "text_color_bgr": self.text_color_swatch,
            "shadow_color_bgr": self.color_swatch,
            "box_color_bgr": self.box_color_swatch,
            "key_color_bgr": self.key_color_swatch,
        }
        for attr, swatch in color_swatches.items():
            color = preset.get(attr)
            if color:
                b, g, r = (int(c) for c in color)
                setattr(self.live_cfg, attr, (b, g, r))
                swatch.config(bg=self._bgr_to_hex((b, g, r)))
        if "box_enabled" in preset:
            self.live_cfg.box_enabled = bool(preset["box_enabled"])
            self.box_enabled_var.set(self.live_cfg.box_enabled)
        if "key_mode" in preset:
            self._set_key_mode(preset["key_mode"])
        self._apply_output_values(preset)
        self._update_source_card()
        self._update_text_card()
        self._update_background_card()
        self._update_output_card()
        app = preset.get("_app")
        if not isinstance(app, dict):
            self._append_log(f"Preset {slot} applied (look only — saved before presets held the full setup).")
        elif self.worker and self.worker.is_alive():
            # Changing the input/voice mid-broadcast would cut the signal: the
            # look is live now, the rest waits for the next Start.
            self._pending_app_settings = app
            self._append_log(
                f"Preset {slot}: look applied. Input, recognition, voice and output name "
                "will apply when you Stop (they take effect on the next Start)."
            )
        else:
            self._apply_app_settings(app)
            self._append_log(f"Preset {slot} applied.")

    def _apply_output_values(self, values):
        """Applies output values (resolution/fps/scale/position) found in
        `values` (OUTPUT_ATTRS keys) to the UI and to output_cfg. Missing keys
        are left unchanged (backward compatible with presets saved before this
        feature existed)."""
        if "canvas_w" in values and "canvas_h" in values:
            w, h = int(values["canvas_w"]), int(values["canvas_h"])
            self.out_width_var.set(w)
            self.out_height_var.set(h)
            self._apply_output_resolution()
            self.resolution_var.set(self._match_resolution_label(w, h))
        if "fps_n" in values and "fps_d" in values:
            self.output_cfg.fps_n = int(values["fps_n"])
            self.output_cfg.fps_d = int(values["fps_d"])
            self.fps_var.set(self._match_fps_label(self.output_cfg.fps_n, self.output_cfg.fps_d))
            self._update_output_info_label()
        for attr in ("scale", "pos_x_px", "pos_y_px"):
            if attr in values and attr in self.sliders:
                self.sliders[attr].set(values[attr])

    # ---------- Mute ----------

    def _toggle_mute(self, event=None):
        if event is not None and isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Text)):
            return
        self.muted = not self.muted
        if self.voice_tuning is not None:
            self.voice_tuning.muted = self.muted
        if self.muted:
            self.mute_btn.set_variant("muted")
            self._append_log("Mute enabled — NDI output is now fully transparent and the voice is silent.")
        else:
            self.mute_btn.set_variant("idle")
            self._append_log("Mute disabled.")

    # ---------- NDI sources ----------

    def _refresh_sources(self):
        # Discovery runs continuously in the background (see
        # _sources_worker_loop): the source list is refreshed automatically
        # as new sources announce themselves, so there is nothing extra to
        # trigger here besides reassuring the user something is happening.
        self.msg_queue.put(("log", "Searching for NDI sources..."))

    def _sources_worker_loop(self):
        """Runs for the whole app session on a single, persistent NDI
        Finder, so sources that take a few seconds to announce themselves
        (different subnet, just started, ...) still show up instead of being
        missed by a one-shot, short discovery window."""
        while not self._sources_stop_event.is_set():
            try:
                sources = self._source_finder.poll(1500)
            except Exception as e:
                self.msg_queue.put(("log", f"Error searching for sources: {e}"))
                self._sources_stop_event.wait(2.0)
                continue
            self.msg_queue.put(("sources", sources))

    # ---------- Start/Stop ----------

    def _toggle_start_stop(self):
        if self.worker and self.worker.is_alive():
            self._stop()
        else:
            self._start()

    def _set_running_visual(self, running):
        self.start_btn.set_variant("running" if running else "idle")

    def _start(self):
        if self.worker and self.worker.is_alive():
            return
        output_name = self.output_name_var.get().strip() or APP_NAME
        if self.input_mode == "audio":
            settings = self._speech_settings()
            if settings is None:
                label = self.audio_device_var.get()
                if label.startswith(NDI_INPUT_PREFIX):
                    self._append_log(f"NDI source '{label[len(NDI_INPUT_PREFIX):]}' isn't on the network (yet).")
                else:
                    self._append_log("Select an audio input device before starting.")
                return
            voice_settings = self._voice_settings()
            if voice_settings is not None and voice_settings.lang != settings.target_lang:
                settings.extra_target_langs = (voice_settings.lang,)
            target, args = self._run_audio_pipeline, (settings, voice_settings, output_name)
        else:
            idx = self.source_combo.current()
            if idx < 0 or idx >= len(self.sources):
                if self._wanted_ndi_source:
                    self._append_log(f"NDI source '{self._wanted_ndi_source}' isn't on the network (yet).")
                else:
                    self._append_log("Select an NDI source before starting.")
                return
            target, args = self._run_pipeline, (self.sources[idx], output_name)

        self._stop_monitor()  # the engine opens its own capture of the device
        self.stop_event.clear()
        self.source_combo.config(state="disabled")
        self._set_input_settings_locked(True)
        self.status_var.set("Starting...")
        self._set_running_visual(True)

        self.worker = threading.Thread(target=target, args=args, daemon=True)
        self.worker.start()

    def _stop(self):
        self.stop_event.set()
        self.status_var.set("Stopping...")
        self._set_running_visual(False)

    def _run_pipeline(self, source, output_name):
        try:
            receiver = ndi_io.Receiver(source)
            sender = ndi_io.Sender(output_name)
        except Exception as e:
            self.msg_queue.put(("log", f"Failed to start: {e}"))
            self.msg_queue.put(("stopped", None))
            return

        self.msg_queue.put(("log", f"Receiving '{source.ndi_name}' -> sending '{output_name}'."))
        self.msg_queue.put(("status", "Running."))

        frames = 0
        last_report = time.time()
        last_preview = 0.0
        last_info = 0.0
        try:
            while not self.stop_event.is_set():
                frame = receiver.read(timeout_ms=500)
                if frame is None:
                    continue
                out = process_frame(frame, self.live_cfg)
                out_cfg = self.output_cfg
                canvas = place_on_canvas(
                    out, out_cfg.canvas_w, out_cfg.canvas_h,
                    out_cfg.scale, out_cfg.pos_x_px, out_cfg.pos_y_px,
                )

                muted_now = self.muted
                out_send = np.zeros_like(canvas) if muted_now else canvas
                sender.send(out_send, frame_rate_n=out_cfg.fps_n, frame_rate_d=out_cfg.fps_d)
                frames += 1

                now = time.time()
                if now - last_preview >= PREVIEW_INTERVAL_S:
                    # The preview always shows the output canvas result, even when
                    # muted (only the NDI output actually sent becomes transparent).
                    # The raw (pre-key) frame rides along so the eyedropper can
                    # sample the true, unprocessed background color.
                    self.msg_queue.put(("preview", (canvas.copy(), muted_now, frame[:, :, :3].copy())))
                    last_preview = now
                if now - last_info >= INFO_INTERVAL_S:
                    self.msg_queue.put(
                        ("input_info", (receiver.last_xres, receiver.last_yres, receiver.last_frame_rate))
                    )
                    last_info = now
                if now - last_report >= 2.0:
                    fps = frames / (now - last_report)
                    self.msg_queue.put(("status", f"Running — {fps:.1f} fps"))
                    frames = 0
                    last_report = now
        except Exception as e:
            self.msg_queue.put(("log", f"Error during processing: {e}"))
        finally:
            receiver.close()
            sender.close()
            self.msg_queue.put(("log", "Stopped."))
            self.msg_queue.put(("stopped", None))

    def _run_audio_pipeline(self, settings, voice_settings, output_name):
        """Audio mode: the speech engine (own thread) feeds a CaptionBoard;
        this loop renders the visible caption lines, runs them through the
        same keyer/shadow/box and output placement as the NDI mode, and
        sends a frame per output frame period. Captions are only re-rendered
        when the text or a setting changes. If a voice is configured, each
        final sentence is also spoken (VoiceOutput runs its own threads)."""
        post = self.msg_queue.put
        try:
            # Paced by this loop, not by the SDK (see ndi_io.Sender).
            sender = ndi_io.Sender(output_name, clock_video=False)
        except Exception as e:
            post(("log", f"Failed to start: {e}"))
            post(("stopped", None))
            return

        engine_status = ["Starting speech engine..."]

        def on_status(text):
            engine_status[0] = text
            post(("status", text))

        board = CaptionBoard()
        voice_out = None
        if voice_settings is not None:
            voice_out = tts.VoiceOutput(voice_settings, self.voice_tuning, sender, log=lambda m: post(("log", m)))
            try:
                voice_out.start()
            except Exception as e:
                post(("log", f"Voice output failed to start: {e}"))
                voice_out = None
            else:
                post(("voice_started", voice_out))

        def on_caption(text, final, info=None):
            if final or self.caption_cfg.live:
                board.update(text, final)
            if final and info and info.get("source") and voice_out is not None:
                voice_out.say(info)

        engine = speech.create_engine(settings, on_caption, log=lambda m: post(("log", m)), on_status=on_status)
        engine.start()
        post(("engine_started", engine))
        post(("log", f"Captioning audio -> sending '{output_name}'."))

        last_key = None
        keyed = None
        frames = 0
        last_report = time.time()
        last_preview = 0.0
        last_level = 0.0
        next_t = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                out_cfg = self.output_cfg
                cap = self.caption_cfg
                w, h = int(out_cfg.canvas_w), int(out_cfg.canvas_h)
                font = load_font(cap.font_size, cap.bold)
                lines = board.visible_lines(
                    font, w * cap.line_width_pct / 100.0, cap.max_lines, cap.hold_s, pop_on=not cap.live
                )
                key = (lines, w, h, astuple(cap), astuple(self.live_cfg))
                if key != last_key:
                    keyed = render_keyed_captions(lines, w, h, cap, font, self.live_cfg)
                    last_key = key
                canvas = place_on_canvas(keyed, w, h, out_cfg.scale, out_cfg.pos_x_px, out_cfg.pos_y_px)

                muted_now = self.muted
                out_send = np.zeros_like(canvas) if muted_now else canvas
                sender.send(out_send, frame_rate_n=out_cfg.fps_n, frame_rate_d=out_cfg.fps_d)
                frames += 1

                # Absolute schedule; the waits let the other threads run.
                next_t += out_cfg.fps_d / max(1, out_cfg.fps_n)
                delay = next_t - time.perf_counter()
                if delay > 0:
                    self.stop_event.wait(delay)
                elif delay < -0.5:
                    next_t = time.perf_counter()

                now = time.time()
                if now - last_preview >= PREVIEW_INTERVAL_S:
                    post(("preview", (canvas.copy(), muted_now, None)))
                    last_preview = now
                if now - last_level >= LEVEL_INTERVAL_S:
                    post(("audio_level", engine.level_db))
                    last_level = now
                if now - last_report >= 2.0:
                    fps = frames / (now - last_report)
                    post(("status", f"{engine_status[0]} — {fps:.1f} fps"))
                    frames = 0
                    last_report = now
        except Exception as e:
            post(("log", f"Error during processing: {e}"))
        finally:
            engine.stop()
            if voice_out is not None:
                voice_out.stop()
            sender.close()
            post(("audio_level", -120.0))
            post(("log", "Stopped."))
            post(("stopped", None))

    # ---------- Message queue (worker thread -> UI) ----------

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "sources":
                    self._set_sources(payload)
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "preview":
                    frame, muted, raw_bgr = payload
                    self.last_raw_bgr = raw_bgr
                    if self.picking_color:
                        self._render_picking_preview()
                    else:
                        self._update_preview(frame, muted)
                elif kind == "input_info":
                    xres, yres, fr = payload
                    if xres and yres:
                        fps_s = self._fmt_fps(*fr) if fr and fr[1] else "—"
                        self.input_info_var.set(f"Input: {xres}x{yres} @ {fps_s} fps")
                        self._update_source_card()
                elif kind == "audio_level":
                    self._update_level_meter(payload)
                elif kind == "voice_started":
                    self._voice_out = payload
                    self._sync_voice_listen()
                elif kind == "engine_started":
                    self._engine = payload
                    self._sync_engine_listen()
                elif kind == "stopped":
                    # The engine has stopped (its capture is closed): listening
                    # moves back to the level monitor, reopened below.
                    self._engine = None
                    self._close_listen_output()
                    self._voice_out = None
                    self._close_voice_listen_output()
                    self._set_running_visual(False)
                    self.source_combo.config(state="readonly")
                    self._set_input_settings_locked(False)
                    self.status_var.set("Stopped.")
                    self.input_info_var.set("Input: —")
                    self._update_source_card()
                    if self._pending_app_settings is not None:
                        pending, self._pending_app_settings = self._pending_app_settings, None
                        self._apply_app_settings(pending)
                        self._append_log("Preset setup (input, recognition, voice, output name) applied.")
                    self._restart_monitor()
                elif kind == "update_checked":
                    status, detail = payload
                    self._on_update_checked(status, detail)
                elif kind == "update_applied":
                    status, detail = payload
                    self._on_update_applied(status, detail)
                elif kind == "requirements_installed":
                    self._on_requirements_installed()
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    def _set_sources(self, sources):
        previous_idx = self.source_combo.current()
        selected_name = (
            self.sources[previous_idx].ndi_name
            if 0 <= previous_idx < len(self.sources)
            else None
        )
        previous_names = {s.ndi_name for s in self.sources}

        self.sources = sources
        names = [s.ndi_name for s in sources]
        self.source_combo["values"] = names

        if self._wanted_ndi_source in names:
            # A source chosen by a preset / the last session, now on the network.
            self.source_combo.current(names.index(self._wanted_ndi_source))
            self._wanted_ndi_source = None
        elif selected_name in names:
            self.source_combo.current(names.index(selected_name))
        elif names and self.source_combo.current() < 0 and not self._wanted_ndi_source:
            self.source_combo.current(0)

        if {s.ndi_name for s in sources} != previous_names:
            self._append_log(f"{len(names)} source(s) found.")
            self._on_sources_changed()
        self._update_source_card()

    def _append_log(self, text):
        self.log_lines.append(text)
        if len(self.log_lines) > MAX_LOG_LINES:
            self.log_lines = self.log_lines[-MAX_LOG_LINES:]
        if self.log_text is None or not self.log_text.winfo_exists():
            return
        self.log_text.config(state="normal")
        self.log_text.insert("end", text + "\n")
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > MAX_LOG_LINES:
            self.log_text.delete("1.0", f"{line_count - MAX_LOG_LINES}.0")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _open_log_window(self):
        if self.log_window is not None and self.log_window.winfo_exists():
            self.log_window.deiconify()
            self.log_window.lift()
            return
        self._build_log_window()

    def _close_log_window(self):
        if self.log_window is not None:
            self.log_window.destroy()
        self.log_window = None
        self.log_text = None

    def _build_log_window(self):
        win = tk.Toplevel(self.root)
        win.title("Log")
        win.geometry("560x320")
        win.minsize(320, 200)
        win.configure(bg=BG)
        self._darken_titlebar(win)
        win.protocol("WM_DELETE_WINDOW", self._close_log_window)
        self.log_window = win

        text = tk.Text(
            win, wrap="word", bg=PANEL_BG, fg=FG, insertbackground=FG,
            relief="flat", borderwidth=0, highlightthickness=1, highlightbackground=BORDER,
            padx=8, pady=8,
        )
        text.pack(fill="both", expand=True, padx=8, pady=8)
        text.insert("end", "\n".join(self.log_lines) + ("\n" if self.log_lines else ""))
        text.see("end")
        text.config(state="disabled")
        self.log_text = text

    def _update_preview(self, bgra, muted):
        if self.preview_canvas is None or not self.preview_canvas.winfo_exists():
            return
        h, w = bgra.shape[:2]
        base_scale = min(1.0, PREVIEW_MAX_W / w)
        total_scale = base_scale * self.zoom
        if total_scale != 1.0:
            new_w = max(1, int(w * total_scale))
            new_h = max(1, int(h * total_scale))
            interp = cv2.INTER_AREA if total_scale < 1.0 else cv2.INTER_LINEAR
            bgra = cv2.resize(bgra, (new_w, new_h), interpolation=interp)
            h, w = bgra.shape[:2]

        bg = np.full((h, w, 3), 90, dtype=np.float32)
        alpha = bgra[:, :, 3:4].astype(np.float32) / 255.0
        rgb = bgra[:, :, [2, 1, 0]].astype(np.float32)
        comp = np.clip(rgb * alpha + bg * (1 - alpha), 0, 255).astype(np.uint8)

        img = Image.fromarray(comp, mode="RGB")
        self.preview_photo = ImageTk.PhotoImage(img)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, image=self.preview_photo, anchor="nw")
        self.preview_canvas.config(scrollregion=(0, 0, w, h))
        if muted:
            self.preview_canvas.create_text(
                12, 10, text="MUTED", anchor="nw", fill=REC_BG,
                font=(self.font_family, 16, "bold"),
            )

    # ---------- Updates (git pull from GitHub) ----------

    def _check_for_updates(self):
        self._append_log("Checking for updates...")
        threading.Thread(target=self._update_check_worker, daemon=True).start()

    def _update_check_worker(self):
        try:
            self._run_git("fetch", "--quiet", GIT_REMOTE, GIT_BRANCH)
            local = self._run_git("rev-parse", "HEAD").strip()
            remote = self._run_git("rev-parse", f"{GIT_REMOTE}/{GIT_BRANCH}").strip()
        except (OSError, subprocess.SubprocessError) as e:
            self.msg_queue.put(("update_checked", ("error", str(e))))
            return
        status = "up_to_date" if local == remote else "available"
        self.msg_queue.put(("update_checked", (status, None)))

    def _on_update_checked(self, status, detail):
        if status == "error":
            self._append_log(f"Update check failed: {detail}")
            messagebox.showerror(f"{APP_NAME} - update", f"Could not check for updates:\n\n{detail}")
        elif status == "up_to_date":
            self._append_log("Already up to date.")
            messagebox.showinfo(f"{APP_NAME} - update", "You're already on the latest version.")
        elif status == "available":
            self._append_log("An update is available.")
            if messagebox.askyesno(
                f"{APP_NAME} - update", "An update is available. Download and apply it now?"
            ):
                self._append_log("Applying update...")
                threading.Thread(target=self._update_apply_worker, daemon=True).start()

    def _update_apply_worker(self):
        try:
            self._run_git("pull", "--ff-only", GIT_REMOTE, GIT_BRANCH)
        except (OSError, subprocess.SubprocessError) as e:
            self.msg_queue.put(("update_applied", ("error", str(e))))
            return
        _installed, error = self._install_requirements()
        self.msg_queue.put(("update_applied", ("ok", error)))

    def _startup_requirements_worker(self):
        """At startup, catches up on libraries an update brought without
        installing them (e.g. pulled by an older version of the app, or by
        hand with 'git pull')."""
        installed, error = self._install_requirements()
        if error:
            self.msg_queue.put(("log", error))
        elif installed and speech is None:
            self.msg_queue.put(("requirements_installed", None))

    def _on_requirements_installed(self):
        if messagebox.askyesno(
            f"{APP_NAME} - update",
            "The libraries for the audio mode were installed. Restart now to use them?",
        ):
            self._restart_app()

    @staticmethod
    def _requirements_stamp_path():
        # Kept in the Python environment itself (the .venv), so a new or
        # recreated environment always gets the requirements installed.
        return Path(sys.prefix) / "alphasubs_requirements.sha1"

    def _install_requirements(self):
        """After an update: installs requirements.txt into the running
        Python environment when it changed since the last install here (or
        was never installed by the app) — a pulled version may need new
        libraries. Returns (installed, error message or None). Not for the
        .exe, which bundles its libraries."""
        if getattr(sys, "frozen", False):
            return False, None
        req = REPO_ROOT / "requirements.txt"
        stamp = self._requirements_stamp_path()
        try:
            digest = hashlib.sha1(req.read_bytes()).hexdigest()
            if stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == digest:
                return False, None
        except OSError as e:
            return False, f"Could not read requirements.txt: {e}"
        post = self.msg_queue.put
        post(("log", "Installing the libraries this version needs (pip) — this can take several minutes..."))
        cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--progress-bar", "off",
               "-r", str(req)]
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace", creationflags=flags,
            )
            tail = []
            for line in proc.stdout:
                line = line.rstrip()
                tail = (tail + [line])[-PIP_LOG_TAIL:]
                if line.startswith(PIP_LOG_PREFIXES):
                    post(("log", f"pip: {line}"))
            proc.wait()
        except OSError as e:
            return False, f"Could not run pip: {e}"
        if proc.returncode != 0:
            for line in tail:
                if not line.startswith(PIP_LOG_PREFIXES):  # those are in the Log already
                    post(("log", f"pip: {line}"))
            return False, (
                "Installing the libraries failed (see Log). Run 'pip install -r requirements.txt' manually."
            )
        try:
            stamp.write_text(digest, encoding="utf-8")
        except OSError:
            pass  # only means pip runs again next time
        post(("log", "Libraries installed."))
        return True, None

    def _on_update_applied(self, status, detail):
        if status == "error":
            self._append_log(f"Update failed: {detail}")
            messagebox.showerror(
                f"{APP_NAME} - update",
                f"Could not apply the update automatically:\n\n{detail}\n\n"
                "You can update manually by running 'git pull' in the project folder.",
            )
            return
        self._append_log("Update applied.")
        if detail:
            self._append_log(detail)
            messagebox.showwarning(f"{APP_NAME} - update", f"The update was applied, but:\n\n{detail}")
        if messagebox.askyesno(f"{APP_NAME} - update", "Update applied. Restart now to use the new version?"):
            self._restart_app()
        else:
            messagebox.showinfo(f"{APP_NAME} - update", "Restart the app manually to use the new version.")

    @staticmethod
    def _run_git(*args):
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise subprocess.SubprocessError(result.stderr.strip() or f"git {' '.join(args)} failed")
        return result.stdout

    def _restart_app(self):
        self._on_close()
        os.execv(sys.executable, [sys.executable] + sys.argv)

    def _on_close(self):
        self._stop_monitor()
        self._save_output_defaults()
        self._save_audio_defaults()
        if self.worker and self.worker.is_alive():
            self.stop_event.set()
            self.worker.join(timeout=4.0)
        self._sources_stop_event.set()
        if self._sources_thread.is_alive():
            self._sources_thread.join(timeout=2.0)
        self._source_finder.close()
        ndi_io.shutdown()
        self.root.destroy()


def main():
    # A --windowed PyInstaller build has no console: sys.stdout/stderr are
    # None, and libraries that print progress (e.g. the Whisper model
    # download) would crash writing to them.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    root = tk.Tk()
    try:
        App(root)
    except Exception as e:
        root.withdraw()
        messagebox.showerror(
            f"{APP_NAME} - startup error",
            f"Could not start the application:\n\n{e}",
        )
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
