"""
Refusal & Quality Analysis with Tkinter Labeling UI and Publication-Quality Graphs.

Processes all steering result directories, provides a GUI for labeling responses
(refusal yes/no + quality 1-10), and generates matplotlib figures for the paper.
"""

import json
import os
import re
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "steering results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

CATEGORIES = ["positive", "negative", "jailbreak", "positive_jailbreak", "negative_jailbreak"]
DEFAULT_MODEL = "12b"


def _discover_models():
    """Return a list of available model names by scanning RESULTS_DIR.

    The root-level category dirs are treated as the default model ('12b').
    Any other subdirectory that itself contains category subdirs is a separate model.
    """
    models = []
    if not os.path.isdir(RESULTS_DIR):
        return models
    # Check if root has any category dirs → default model
    if any(os.path.isdir(os.path.join(RESULTS_DIR, c)) and c not in (".",)
           for c in os.listdir(RESULTS_DIR)
           if c in CATEGORIES):
        models.append(DEFAULT_MODEL)
    # Check for model subdirs (e.g. "4b") that contain category subdirs
    for name in sorted(os.listdir(RESULTS_DIR)):
        subdir = os.path.join(RESULTS_DIR, name)
        if not os.path.isdir(subdir) or name in CATEGORIES:
            continue
        if any(os.path.isdir(os.path.join(subdir, c)) for c in CATEGORIES):
            models.append(name)
    return models


def _model_results_dir(model):
    """Return the results directory for a given model."""
    if model == DEFAULT_MODEL:
        return RESULTS_DIR
    return os.path.join(RESULTS_DIR, model)


def _labels_file(model):
    """Return the labels JSON path for a given model."""
    if model == DEFAULT_MODEL:
        return os.path.join(BASE_DIR, "labels.json")
    return os.path.join(BASE_DIR, f"labels_{model}.json")


def _csv_output(model):
    """Return the CSV output path for a given model."""
    if model == DEFAULT_MODEL:
        return os.path.join(BASE_DIR, "final_analysis.csv")
    return os.path.join(BASE_DIR, f"final_analysis_{model}.csv")


# ──────────────────────────────────────────────────────────────
# 1. Parse all results
# ──────────────────────────────────────────────────────────────

def parse_coefficients(filename):
    """Extract coefficient values from a filename.

    Examples:
        pos_0.5_results.json          → {"pos": 0.5}
        neg_-1.0_jb_-0.5_results.json → {"neg": -1.0, "jb": -0.5}
        jb_0.5_results.json           → {"jb": 0.5}
    """
    stem = filename.replace("_results.json", "")
    coeffs = {}
    for m in re.finditer(r"(pos|neg|jb)_([-]?\d+\.?\d*)", stem):
        coeffs[m.group(1)] = float(m.group(2))
    return coeffs


def parse_all_results(model=DEFAULT_MODEL):
    """Load every JSON file across all category subdirectories for a model.

    Returns a list of dicts, one per (file, entry_index) pair:
        {category, filename, entry_idx, original_prompt, steered_response,
         config, coefficients, avg_metrics}
    """
    results_dir = _model_results_dir(model)
    entries = []
    for category in CATEGORIES:
        cat_dir = os.path.join(results_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        for fname in sorted(os.listdir(cat_dir)):
            if not fname.endswith("_results.json"):
                continue
            fpath = os.path.join(cat_dir, fname)
            with open(fpath, "r") as f:
                data = json.load(f)
            coeffs = parse_coefficients(fname)
            for idx, entry in enumerate(data):
                # Average the per-token steering metrics for this entry
                metrics = entry.get("steering_metrics", [])
                avg_metrics = {}
                if metrics:
                    df_m = pd.DataFrame(metrics)
                    avg_metrics = df_m.mean().to_dict()

                entries.append({
                    "category": category,
                    "filename": fname,
                    "entry_idx": idx,
                    "original_prompt": entry.get("original_prompt", ""),
                    "steered_response": entry.get("steered_response", ""),
                    "config": entry.get("config", ""),
                    "coefficients": coeffs,
                    "avg_metrics": avg_metrics,
                })
    return entries


# ──────────────────────────────────────────────────────────────
# 2. Tkinter Labeling UI
# ──────────────────────────────────────────────────────────────

class LabelerApp(tk.Tk):
    # ── Color theme (light, high-contrast) ───────────────────
    BG = "#f5f5f7"           # light warm grey background
    BG_CARD = "#ffffff"      # white card panels
    FG = "#1d1d1f"           # near-black main text
    FG_DIM = "#86868b"       # secondary/muted text
    ACCENT = "#0071e3"       # Apple-blue accent
    GREEN = "#34c759"
    RED = "#ff3b30"
    YELLOW = "#ff9f0a"
    BORDER = "#d2d2d7"
    BTN_BG = "#e8e8ed"
    BTN_ACTIVE = "#d1d1d6"
    BTN_FG = "#1d1d1f"       # always readable on light buttons

    DEFAULT_FONT_SIZE = 13
    FONT_FAMILY = "SF Pro"   # macOS; _resolve_font picks fallback

    def __init__(self, entries, model=DEFAULT_MODEL, available_models=None):
        super().__init__()
        self.model = model
        self.available_models = available_models or [model]
        self.title(f"Steering Response Labeler — {model}")
        self.geometry("1060x820")
        self.minsize(800, 650)
        self.configure(bg=self.BG)

        self.entries = entries
        self.labels_file = _labels_file(model)
        self.labels = self._load_labels()
        self.font_size = self.DEFAULT_FONT_SIZE
        self.font_family = self._resolve_font()
        self.current = 0
        self._jump_to_next_unlabeled()

        # Track all "scalable" widgets so _update_fonts can resize everything
        self._scalable_widgets = []

        self._configure_styles()
        self._build_ui()
        self._bind_keys()
        self._show_current()

    # ── font resolution ──────────────────────────────────────

    def _resolve_font(self):
        """Pick the best available font family for this platform."""
        import tkinter.font as tkfont
        available = set(tkfont.families())
        for candidate in ("SF Pro", "SF Pro Text", "SF Pro Display",
                          ".AppleSystemUIFont",
                          "Segoe UI",       # Windows
                          "Ubuntu", "Cantarell",  # Linux
                          "Helvetica Neue", "Helvetica"):
            if candidate in available:
                return candidate
        return "TkDefaultFont"

    # ── theming ──────────────────────────────────────────────

    def _configure_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TProgressbar", troughcolor=self.BORDER,
                         background=self.ACCENT, thickness=8)

    def _font(self, size=None, bold=False):
        s = size or self.font_size
        return (self.font_family, s, "bold" if bold else "normal")

    # ── persistence ──────────────────────────────────────────

    @staticmethod
    def _label_key(entry):
        return f"{entry['category']}/{entry['filename']}/{entry['entry_idx']}"

    def _load_labels(self):
        if os.path.exists(self.labels_file):
            with open(self.labels_file, "r") as f:
                return json.load(f)
        return {}

    def _save_labels(self):
        with open(self.labels_file, "w") as f:
            json.dump(self.labels, f, indent=2)

    # ── navigation helpers ───────────────────────────────────

    def _jump_to_next_unlabeled(self):
        for i in range(len(self.entries)):
            if self._label_key(self.entries[i]) not in self.labels:
                self.current = i
                return
        self.current = len(self.entries) - 1

    # ── keyboard shortcuts ───────────────────────────────────

    def _bind_keys(self):
        self.bind("<Return>", lambda _: self._go_next())
        self.bind("<Right>", lambda _: self._go_next())
        self.bind("<Left>", lambda _: self._go_back())
        self.bind("<y>", lambda _: self._set_refusal("yes"))
        self.bind("<n>", lambda _: self._set_refusal("no"))
        self.bind("<Command-equal>", lambda _: self._change_font_size(1))
        self.bind("<Command-minus>", lambda _: self._change_font_size(-1))
        self.bind("<Control-equal>", lambda _: self._change_font_size(1))
        self.bind("<Control-minus>", lambda _: self._change_font_size(-1))
        for digit in range(10):
            val = 10 if digit == 0 else digit
            self.bind(str(digit), lambda _, v=val: self._set_quality(v))

    # ── font size (scales ALL text) ──────────────────────────

    def _change_font_size(self, delta):
        self.font_size = max(9, min(26, self.font_size + delta))
        self._update_fonts()

    def _update_fonts(self):
        s = self.font_size
        # Text areas
        self.prompt_text.config(font=self._font(s))
        self.response_text.config(font=self._font(s))
        # All tracked scalable widgets (Labels use .config, pill buttons use _btn_redraw)
        for widget, size_offset, bold in self._scalable_widgets:
            if hasattr(widget, "_draw_pill"):
                self._btn_redraw(widget)
            else:
                try:
                    widget.config(font=self._font(s + size_offset, bold))
                except tk.TclError:
                    pass
        self.font_size_label.config(text=f"{s}pt")

    def _track(self, widget, size_offset=0, bold=False):
        """Register a widget so its font scales with Cmd+/-."""
        self._scalable_widgets.append((widget, size_offset, bold))
        return widget

    # ── pill button (canvas-drawn rounded rect) ────────────────

    @staticmethod
    def _hex_lerp(c1, c2, t):
        """Linearly interpolate between two hex colors."""
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _btn(self, parent, text, command, bg=None, fg=None,
             size_offset=0, bold=False, padx=18, pady=8, radius=14):
        """Create a pill-shaped button drawn on a Canvas with hover/press effects."""
        bg = bg or self.BTN_BG
        fg = fg or self.BTN_FG
        hover_bg = self._hex_lerp(bg, "#ffffff", 0.18)
        press_bg = self._hex_lerp(bg, "#000000", 0.10)

        # Use a Frame as the container so .pack()/.grid() work normally
        frame = tk.Frame(parent, bg=self.BG, cursor="hand2")

        font_tuple = self._font(self.font_size + size_offset, bold)

        canvas = tk.Canvas(frame, highlightthickness=0, bd=0, bg=self.BG)
        canvas.pack()

        # We need to measure the text to size the canvas
        tmp_id = canvas.create_text(0, 0, text=text, font=font_tuple, anchor="nw")
        bbox = canvas.bbox(tmp_id)
        canvas.delete(tmp_id)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        w = tw + padx * 2
        h = th + pady * 2
        canvas.config(width=w, height=h)

        # Draw rounded rectangle
        def _draw_pill(fill):
            canvas.delete("pill")
            r = min(radius, h // 2)
            canvas.create_arc(0, 0, r * 2, r * 2, start=90, extent=90,
                              fill=fill, outline=fill, tags="pill")
            canvas.create_arc(w - r * 2, 0, w, r * 2, start=0, extent=90,
                              fill=fill, outline=fill, tags="pill")
            canvas.create_arc(0, h - r * 2, r * 2, h, start=180, extent=90,
                              fill=fill, outline=fill, tags="pill")
            canvas.create_arc(w - r * 2, h - r * 2, w, h, start=270, extent=90,
                              fill=fill, outline=fill, tags="pill")
            canvas.create_rectangle(r, 0, w - r, h, fill=fill, outline=fill, tags="pill")
            canvas.create_rectangle(0, r, w, h - r, fill=fill, outline=fill, tags="pill")

        _draw_pill(bg)
        text_id = canvas.create_text(w // 2, h // 2, text=text, font=font_tuple,
                                      fill=fg, anchor="center", tags="label")

        # Store state on the frame so we can update colors later
        frame._pill_bg = bg
        frame._pill_fg = fg
        frame._pill_hover_bg = hover_bg
        frame._pill_press_bg = press_bg
        frame._draw_pill = _draw_pill
        frame._canvas = canvas
        frame._text_id = text_id
        frame._font_tuple = font_tuple
        frame._command = command
        frame._size_offset = size_offset
        frame._bold = bold
        frame._padx = padx
        frame._pady = pady
        frame._radius = radius
        frame._text = text

        # Events
        def on_enter(_):
            _draw_pill(frame._pill_hover_bg)
            canvas.tag_raise("label")

        def on_leave(_):
            _draw_pill(frame._pill_bg)
            canvas.tag_raise("label")

        def on_press(_):
            _draw_pill(frame._pill_press_bg)
            canvas.tag_raise("label")

        def on_release(_):
            _draw_pill(frame._pill_hover_bg)
            canvas.tag_raise("label")
            command()

        for target in (canvas, frame):
            target.bind("<Enter>", on_enter)
            target.bind("<Leave>", on_leave)
            target.bind("<ButtonPress-1>", on_press)
            target.bind("<ButtonRelease-1>", on_release)

        # Register for font scaling
        self._scalable_widgets.append((frame, size_offset, bold))
        return frame

    def _btn_recolor(self, frame, bg, fg):
        """Change a pill button's colors (used for selection highlights)."""
        frame._pill_bg = bg
        frame._pill_fg = fg
        frame._pill_hover_bg = self._hex_lerp(bg, "#ffffff", 0.18)
        frame._pill_press_bg = self._hex_lerp(bg, "#000000", 0.10)
        frame._draw_pill(bg)
        frame._canvas.tag_raise("label")
        frame._canvas.itemconfig(frame._text_id, fill=fg)

    def _btn_redraw(self, frame):
        """Redraw a pill button after font size change."""
        canvas = frame._canvas
        font_tuple = self._font(self.font_size + frame._size_offset, frame._bold)
        frame._font_tuple = font_tuple

        # Re-measure text
        canvas.itemconfig(frame._text_id, font=font_tuple)
        tmp_id = canvas.create_text(0, 0, text=frame._text, font=font_tuple, anchor="nw")
        bbox = canvas.bbox(tmp_id)
        canvas.delete(tmp_id)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        w = tw + frame._padx * 2
        h = th + frame._pady * 2
        canvas.config(width=w, height=h)
        frame._draw_pill(frame._pill_bg)
        canvas.coords(frame._text_id, w // 2, h // 2)
        canvas.tag_raise("label")

    # ── UI construction ──────────────────────────────────────

    def _build_ui(self):
        PAD = 18

        # ── Top bar: model selector + progress label + font size controls ─
        top = tk.Frame(self, bg=self.BG)
        top.pack(fill="x", padx=PAD, pady=(12, 0))

        # Model selector
        if len(self.available_models) > 1:
            model_frame = tk.Frame(top, bg=self.BG)
            model_frame.pack(side="left", padx=(0, 16))
            self._track(
                tk.Label(model_frame, text="Model:", font=self._font(self.font_size - 2, bold=True),
                         bg=self.BG, fg=self.FG), size_offset=-2, bold=True
            ).pack(side="left", padx=(0, 6))
            for m in self.available_models:
                is_current = (m == self.model)
                bg = self.ACCENT if is_current else self.BTN_BG
                fg = "#ffffff" if is_current else self.BTN_FG
                b = self._btn(model_frame, f" {m} ",
                              lambda model=m: self._switch_model(model),
                              bg=bg, fg=fg, bold=is_current,
                              size_offset=-2, padx=10, pady=4)
                b.pack(side="left", padx=2)

        self.progress_label = self._track(
            tk.Label(top, text="", font=self._font(self.font_size - 2),
                     bg=self.BG, fg=self.FG_DIM), size_offset=-2)
        self.progress_label.pack(side="left")

        # Font controls
        font_frame = tk.Frame(top, bg=self.BG)
        font_frame.pack(side="right")
        self._track(
            tk.Label(font_frame, text="Text size:", font=self._font(self.font_size - 3),
                     bg=self.BG, fg=self.FG_DIM), size_offset=-3
        ).pack(side="left", padx=(0, 4))
        self._btn(font_frame, " A- ", lambda: self._change_font_size(-1),
                  size_offset=-2, padx=6, pady=2).pack(side="left", padx=1)
        self.font_size_label = self._track(
            tk.Label(font_frame, text=f"{self.font_size}pt",
                     font=self._font(self.font_size - 3),
                     bg=self.BG, fg=self.FG_DIM, width=4), size_offset=-3)
        self.font_size_label.pack(side="left")
        self._btn(font_frame, " A+ ", lambda: self._change_font_size(1),
                  size_offset=-2, padx=6, pady=2).pack(side="left", padx=1)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(self, variable=self.progress_var, maximum=100,
                         style="TProgressbar").pack(fill="x", padx=PAD, pady=(6, 2))

        # Info / category line
        self.info_var = tk.StringVar()
        self._track(
            tk.Label(self, textvariable=self.info_var,
                     font=self._font(self.font_size - 1),
                     bg=self.BG, fg=self.ACCENT), size_offset=-1
        ).pack(anchor="w", padx=PAD, pady=(2, 0))

        # ── Prompt card ──────────────────────────────────────
        self._track(
            tk.Label(self, text="Original Prompt",
                     font=self._font(self.font_size - 1, bold=True),
                     bg=self.BG, fg=self.FG_DIM), size_offset=-1, bold=True
        ).pack(anchor="w", padx=PAD, pady=(10, 3))

        prompt_frame = tk.Frame(self, bg=self.BORDER, bd=1, relief="solid")
        prompt_frame.pack(fill="x", padx=PAD)
        self.prompt_text = scrolledtext.ScrolledText(
            prompt_frame, height=3, wrap="word", state="disabled",
            font=self._font(), bg=self.BG_CARD, fg=self.FG,
            insertbackground=self.FG, selectbackground=self.ACCENT,
            relief="flat", bd=8, highlightthickness=0,
        )
        self.prompt_text.pack(fill="x")

        # ── Response card ────────────────────────────────────
        self._track(
            tk.Label(self, text="Steered Response",
                     font=self._font(self.font_size - 1, bold=True),
                     bg=self.BG, fg=self.FG_DIM), size_offset=-1, bold=True
        ).pack(anchor="w", padx=PAD, pady=(10, 3))

        resp_frame = tk.Frame(self, bg=self.BORDER, bd=1, relief="solid")
        resp_frame.pack(fill="both", expand=True, padx=PAD)
        self.response_text = scrolledtext.ScrolledText(
            resp_frame, height=12, wrap="word", state="disabled",
            font=self._font(), bg=self.BG_CARD, fg=self.FG,
            insertbackground=self.FG, selectbackground=self.ACCENT,
            relief="flat", bd=8, highlightthickness=0,
        )
        self.response_text.pack(fill="both", expand=True)

        # ── Status badge ─────────────────────────────────────
        self.status_var = tk.StringVar()
        self.status_label = self._track(
            tk.Label(self, textvariable=self.status_var,
                     font=self._font(self.font_size - 2),
                     bg=self.BG, fg=self.GREEN), size_offset=-2)
        self.status_label.pack(anchor="w", padx=PAD, pady=(6, 0))

        # ── Controls: Refusal row ────────────────────────────
        refusal_row = tk.Frame(self, bg=self.BG)
        refusal_row.pack(pady=(8, 0))

        self._track(
            tk.Label(refusal_row, text="Refusal:", font=self._font(bold=True),
                     bg=self.BG, fg=self.FG), bold=True
        ).pack(side="left", padx=(0, 12))

        self.refusal_var = tk.StringVar(value="")
        self.btn_yes = self._btn(refusal_row, "  Yes  (Y)  ",
                                  lambda: self._set_refusal("yes"),
                                  bold=True, pady=8)
        self.btn_yes.pack(side="left", padx=4)
        self.btn_no = self._btn(refusal_row, "  No  (N)  ",
                                 lambda: self._set_refusal("no"),
                                 bold=True, pady=8)
        self.btn_no.pack(side="left", padx=4)

        # ── Controls: Quality row ────────────────────────────
        quality_row = tk.Frame(self, bg=self.BG)
        quality_row.pack(pady=(6, 0))

        self._track(
            tk.Label(quality_row, text="Quality:", font=self._font(bold=True),
                     bg=self.BG, fg=self.FG), bold=True
        ).pack(side="left", padx=(0, 12))

        self.quality_var = tk.IntVar(value=5)
        self.quality_btns = []
        for i in range(1, 11):
            b = self._btn(quality_row, f" {i} " if i < 10 else "10",
                          lambda v=i: self._set_quality(v),
                          size_offset=-1, padx=8, pady=5)
            b.pack(side="left", padx=2)
            self.quality_btns.append(b)

        # ── Navigation row ───────────────────────────────────
        nav = tk.Frame(self, bg=self.BG)
        nav.pack(pady=(10, 6))

        self.btn_back = self._btn(nav, "  Back  ", self._go_back)
        self.btn_back.pack(side="left", padx=6)

        self.btn_skip = self._btn(nav, "  Skip  ", self._skip)
        self.btn_skip.pack(side="left", padx=6)

        self.btn_next = self._btn(nav, "  Save & Next  ", self._go_next,
                                   bg=self.ACCENT, fg="#ffffff", bold=True, pady=8)
        self.btn_next.pack(side="left", padx=6)

        self.btn_done = self._btn(nav, "  Finish  ", self._finish)
        self.btn_done.pack(side="left", padx=6)

        # ── Shortcut hint ────────────────────────────────────
        self._track(
            tk.Label(self,
                     text="Y / N  refusal    1-9, 0  quality    Enter / \u2192  save & next    \u2190  back    Cmd +/-  text size",
                     font=self._font(self.font_size - 4),
                     bg=self.BG, fg=self.FG_DIM), size_offset=-4
        ).pack(side="bottom", pady=(0, 8))

    # ── display ──────────────────────────────────────────────

    def _show_current(self):
        entry = self.entries[self.current]
        key = self._label_key(entry)

        # Progress
        labeled_count = sum(1 for e in self.entries if self._label_key(e) in self.labels)
        total = len(self.entries)
        pct = labeled_count / total * 100 if total else 0
        self.progress_var.set(pct)
        self.progress_label.config(text=f"Labeled {labeled_count} / {total}  ({pct:.0f}%)")

        # Info
        coeffs_str = ", ".join(f"{k}={v}" for k, v in entry["coefficients"].items())
        self.info_var.set(
            f"[{self.current + 1}/{total}]   {entry['category']}  /  {entry['filename']}   \u2014   {coeffs_str}")

        # Status badge
        if key in self.labels:
            lbl = self.labels[key]
            self.status_var.set(
                f"Already labeled: refusal={lbl['refusal']}, quality={lbl.get('quality', '?')}")
            self.status_label.config(fg=self.GREEN)
        else:
            self.status_var.set("Not yet labeled")
            self.status_label.config(fg=self.FG_DIM)

        # Prompt
        self.prompt_text.config(state="normal")
        self.prompt_text.delete("1.0", "end")
        self.prompt_text.insert("1.0", entry["original_prompt"])
        self.prompt_text.config(state="disabled")

        # Response
        self.response_text.config(state="normal")
        self.response_text.delete("1.0", "end")
        self.response_text.insert("1.0", entry["steered_response"])
        self.response_text.config(state="disabled")
        self.response_text.yview_moveto(0)

        # Restore existing label
        if key in self.labels:
            self._set_refusal(self.labels[key]["refusal"])
            self._set_quality(self.labels[key].get("quality", 5))
        else:
            self._set_refusal("")
            self._set_quality(5)

    def _set_refusal(self, value):
        self.refusal_var.set(value)
        if value == "yes":
            self._btn_recolor(self.btn_yes, self.RED, "#ffffff")
            self._btn_recolor(self.btn_no, self.BTN_BG, self.BTN_FG)
        elif value == "no":
            self._btn_recolor(self.btn_no, self.GREEN, "#ffffff")
            self._btn_recolor(self.btn_yes, self.BTN_BG, self.BTN_FG)
        else:
            self._btn_recolor(self.btn_yes, self.BTN_BG, self.BTN_FG)
            self._btn_recolor(self.btn_no, self.BTN_BG, self.BTN_FG)

    def _set_quality(self, value):
        self.quality_var.set(value)
        for i, btn in enumerate(self.quality_btns):
            if i + 1 == value:
                self._btn_recolor(btn, self.ACCENT, "#ffffff")
            else:
                self._btn_recolor(btn, self.BTN_BG, self.BTN_FG)

    # ── navigation ───────────────────────────────────────────

    def _save_current(self):
        if not self.refusal_var.get():
            messagebox.showwarning("Missing label", "Please select Yes or No for refusal.")
            return False
        entry = self.entries[self.current]
        key = self._label_key(entry)
        self.labels[key] = {
            "refusal": self.refusal_var.get(),
            "quality": self.quality_var.get(),
        }
        self._save_labels()
        return True

    def _go_next(self):
        if not self._save_current():
            return
        if self.current < len(self.entries) - 1:
            self.current += 1
            self._show_current()
        else:
            messagebox.showinfo("Done", "All entries have been reviewed!")

    def _go_back(self):
        if self.current > 0:
            self.current -= 1
            self._show_current()

    def _skip(self):
        if self.current < len(self.entries) - 1:
            self.current += 1
            self._show_current()

    def _finish(self):
        labeled_count = sum(1 for e in self.entries if self._label_key(e) in self.labels)
        total = len(self.entries)
        if labeled_count < total:
            if not messagebox.askyesno("Incomplete",
                                       f"Only {labeled_count}/{total} entries labeled. Finish anyway?"):
                return
        self.destroy()

    def _switch_model(self, model):
        """Switch to a different model's results. Saves current labels first."""
        if model == self.model:
            return
        # Save current work
        self._save_labels()
        # Store new model and relaunch — we set an attribute the main loop checks
        self._next_model = model
        self.destroy()


# ──────────────────────────────────────────────────────────────
# 3. Aggregate results
# ──────────────────────────────────────────────────────────────

def aggregate_results(entries, model=DEFAULT_MODEL):
    """Combine labels + metrics into a per-file DataFrame."""
    lf = _labels_file(model)
    labels = {}
    if os.path.exists(lf):
        with open(lf, "r") as f:
            labels = json.load(f)

    # Group entries by (category, filename)
    from collections import defaultdict
    groups = defaultdict(list)
    for e in entries:
        groups[(e["category"], e["filename"])].append(e)

    rows = []
    for (category, filename), group_entries in sorted(groups.items()):
        total = len(group_entries)
        refusals = 0
        qualities = []
        metrics_acc = []

        for e in group_entries:
            key = f"{e['category']}/{e['filename']}/{e['entry_idx']}"
            lbl = labels.get(key, {})
            if lbl.get("refusal") == "yes":
                refusals += 1
            if "quality" in lbl:
                qualities.append(lbl["quality"])
            if e["avg_metrics"]:
                metrics_acc.append(e["avg_metrics"])

        coeffs = group_entries[0]["coefficients"]
        row = {
            "category": category,
            "filename": filename,
            "config": group_entries[0]["config"],
            "num_samples": total,
            "refusal_rate_pct": (refusals / total) * 100 if total else 0,
            "avg_quality": np.mean(qualities) if qualities else np.nan,
        }
        # Add coefficient columns
        for k, v in coeffs.items():
            row[f"coeff_{k}"] = v

        # Average metrics across entries
        if metrics_acc:
            df_m = pd.DataFrame(metrics_acc)
            for col in df_m.columns:
                row[col] = df_m[col].mean()

        rows.append(row)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# 4. Graph generation
# ──────────────────────────────────────────────────────────────

def _setup_style():
    """Set up a clean academic matplotlib style."""
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        plt.style.use("seaborn-paper")
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    })


def _save_fig(fig, name, fig_dir=None):
    d = fig_dir or FIGURES_DIR
    os.makedirs(d, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    rel = os.path.relpath(d, BASE_DIR)
    print(f"  Saved {rel}/{name}.pdf and .png")


def generate_graphs(df, model=DEFAULT_MODEL):
    """Generate all publication-quality figures."""
    _setup_style()
    fig_dir = os.path.join(FIGURES_DIR, model) if model != DEFAULT_MODEL else FIGURES_DIR
    os.makedirs(fig_dir, exist_ok=True)

    # ── Style helper ──
    def _style_ax(a):
        a.set_facecolor("#fafafa")
        a.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    palette = {
        "positive": "#4c72b0", "negative": "#dd8452", "jailbreak": "#55a868",
        "positive_jailbreak": "#c44e52", "negative_jailbreak": "#8172b3",
    }

    # ── 4c. Reconstruction error to original ratio bar chart ──
    if "recon_error_to_original_ratio" in df.columns:
        df_rer = df.dropna(subset=["recon_error_to_original_ratio"]).sort_values(["category", "config"])
        if not df_rer.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_facecolor("#fafafa")
            ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
            x_labels_rer = df_rer["config"].tolist()
            rer_vals = df_rer["recon_error_to_original_ratio"].tolist()
            colors_rer = [palette.get(r["category"], "#999999") for _, r in df_rer.iterrows()]
            ax.bar(range(len(x_labels_rer)), rer_vals, color=colors_rer,
                   edgecolor="white", linewidth=0.5)
            ax.set_xticks(range(len(x_labels_rer)))
            ax.set_xticklabels(x_labels_rer, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel(r"$\|\mathbf{h} - \mathrm{Dec}(\mathbf{z})\|_2 \,/\, \|\mathbf{h}\|_2$")
            ax.set_title("Reconstruction Error / Original Norm", fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            handles_rer = [Patch(color=palette[c], label=c.replace("_", " ").title())
                           for c in CATEGORIES if c in set(df_rer["category"])]
            ax.legend(handles=handles_rer, loc="upper left", bbox_to_anchor=(1.0, 1.0),
                      frameon=True, fontsize=7)
            fig.tight_layout()
            _save_fig(fig, "recon_error_ratio_bar", fig_dir)

    # ── 5. Bar chart: steering delta / original ratio per config ──
    if "delta_to_original_ratio" in df.columns:
        df_d = df.dropna(subset=["delta_to_original_ratio"]).sort_values(["category", "config"])
        if not df_d.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            x_labels_d = df_d["config"].tolist()
            delta_vals = df_d["delta_to_original_ratio"].tolist()
            colors_d = [palette.get(row["category"], "#999999") for _, row in df_d.iterrows()]
            ax.bar(range(len(x_labels_d)), delta_vals, color=colors_d, edgecolor="white", linewidth=0.5)
            ax.set_xticks(range(len(x_labels_d)))
            ax.set_xticklabels(x_labels_d, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel(r"$\|\Delta\| \,/\, \|\mathbf{h}\|$")
            ax.set_title("Steering Magnitude (Delta / Original Norm)")
            handles_d = [Patch(color=palette[c], label=c.replace("_", " ").title())
                         for c in CATEGORIES if c in set(df_d["category"])]
            ax.legend(handles=handles_d, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=True, fontsize=7)
            fig.tight_layout()
            _save_fig(fig, "delta_ratio_bar", fig_dir)

    # ── 6. Quality distribution for non-refusal responses only ──
    lf = _labels_file(model)
    if os.path.exists(lf):
        with open(lf, "r") as f:
            all_labels = json.load(f)
        non_refusal_q = [v["quality"] for v in all_labels.values()
                         if v.get("refusal") == "no" and "quality" in v]
        refusal_q = [v["quality"] for v in all_labels.values()
                     if v.get("refusal") == "yes" and "quality" in v]
        if non_refusal_q:
            fig, ax = plt.subplots(figsize=(8, 5))
            bins = np.arange(0.5, 11.5, 1)
            if refusal_q:
                ax.hist(refusal_q, bins=bins, alpha=0.5, label="Refusal responses",
                        color="#4c72b0", edgecolor="white")
            ax.hist(non_refusal_q, bins=bins, alpha=0.7, label="Non-refusal (jailbroken)",
                    color="#c44e52", edgecolor="white")
            ax.set_xlabel("Quality Rating")
            ax.set_ylabel("Count")
            ax.set_title("Quality Distribution: Refusal vs. Jailbroken Responses")
            ax.set_xticks(range(1, 11))
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=True, fontsize=8)
            fig.tight_layout()
            _save_fig(fig, "quality_distribution_histogram", fig_dir)

    # ── Helper: annotate points on a line plot, avoiding overlaps ──
    def _annotate_no_overlap(ax, points, fontsize=6):
        """Annotate (x, y, text, color) tuples without overlaps.

        Measures each label's pixel size once, then uses greedy placement
        with actual bbox overlap checks over a dense grid of candidates.
        """
        if not points:
            return

        fig = ax.get_figure()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        pad = 3  # px padding around each bbox

        # Measure text size for each label (place temporarily, read bbox, remove)
        text_sizes = []  # (half_w, half_h) in display px
        for x, y, text, color in points:
            tmp = ax.text(0.5, 0.5, text, fontsize=fontsize, transform=ax.transAxes,
                          bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                    ec="none", alpha=0.7))
            fig.canvas.draw()
            bb = tmp.get_window_extent(renderer)
            text_sizes.append(((bb.width / 2) + pad, (bb.height / 2) + pad))
            tmp.remove()

        # Build dense candidate offsets (in points)
        candidates = []
        for dy in [10, -12, 18, -20, 28, -30, 38, -40, 48, -50]:
            for dx in [0, 10, -10, 20, -20, 30, -30]:
                candidates.append((dx, dy))

        # Convert data points to display coords
        data_display = [ax.transData.transform((x, y)) for x, y, _, _ in points]

        def _bbox(center_x, center_y, hw, hh):
            return (center_x - hw, center_y - hh, center_x + hw, center_y + hh)

        def _overlaps_any(box, placed):
            for pb in placed:
                if box[0] < pb[2] and box[2] > pb[0] and box[1] < pb[3] and box[3] > pb[1]:
                    return True
            return False

        def _total_overlap(box, placed):
            total = 0
            for pb in placed:
                dx = max(0, min(box[2], pb[2]) - max(box[0], pb[0]))
                dy = max(0, min(box[3], pb[3]) - max(box[1], pb[1]))
                total += dx * dy
            return total

        placed_bboxes = []
        for i, (x, y, text, color) in enumerate(points):
            disp_x, disp_y = data_display[i]
            hw, hh = text_sizes[i]

            best_offset = candidates[0]
            best_cost = float("inf")
            for dx, dy in candidates:
                cx, cy = disp_x + dx, disp_y + dy
                box = _bbox(cx, cy, hw, hh)
                ov = _total_overlap(box, placed_bboxes)
                # Also penalise distance from point (prefer closer labels)
                dist_cost = (dx**2 + dy**2) ** 0.5 * 0.1
                cost = ov + dist_cost
                if cost < best_cost:
                    best_cost = cost
                    best_offset = (dx, dy)
                if ov == 0 and abs(dy) <= 12:
                    break  # good enough
            ox, oy = best_offset
            ha = "center" if abs(ox) < 5 else ("left" if ox > 0 else "right")
            ax.annotate(text, (x, y), textcoords="offset points",
                        xytext=(ox, oy), ha=ha, fontsize=fontsize, color=color,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.7))
            # Record placed bbox
            cx, cy = disp_x + ox, disp_y + oy
            placed_bboxes.append(_bbox(cx, cy, hw, hh))

    # ── 8. Combined line plot: refusal rate vs sentiment coefficient ──
    panels = [
        ("positive_jailbreak", "positive", "coeff_pos", "Positive Sentiment Coefficient"),
        ("negative_jailbreak", "negative", "coeff_neg", "Negative Sentiment Coefficient"),
    ]
    panels = [(jb_cat, solo_cat, col, lbl) for jb_cat, solo_cat, col, lbl in panels
              if col in df.columns and not df[(df["category"] == jb_cat) & df[col].notna()].empty]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5.5), squeeze=False)
        fig.patch.set_facecolor("white")
        jb_colors = {-0.75: "#2ca02c", -0.5: "#c44e52", 0.5: "#4c72b0", 0.75: "#8172b3"}
        jb_markers = {-0.75: "D", -0.5: "o", 0.5: "s", 0.75: "^"}
        solo_color = "#55a868"
        solo_marker = "D"
        for idx, (jb_cat, solo_cat, coeff_col, xlabel) in enumerate(panels):
            ax = axes[0][idx]
            _style_ax(ax)
            all_points = []  # collect (x, y, text, color) for deferred annotation
            # Sentiment-only line
            solo = df[(df["category"] == solo_cat) & df[coeff_col].notna()].copy()
            if not solo.empty and len(solo) >= 2:
                solo = solo.sort_values(coeff_col)
                ax.plot(solo[coeff_col], solo["refusal_rate_pct"], f"{solo_marker}--",
                        color=solo_color, linewidth=2, markersize=8,
                        markeredgecolor="white", markeredgewidth=1,
                        label="Sentiment only (no JB)", alpha=0.8)
                for _, row in solo.iterrows():
                    all_points.append((row[coeff_col], row["refusal_rate_pct"],
                                       f"{row['refusal_rate_pct']:.0f}%", solo_color))
            # Sentiment + JB lines
            sub = df[(df["category"] == jb_cat) & df[coeff_col].notna()].copy()
            jb_vals = sorted(sub["coeff_jb"].dropna().unique())
            for jb_v in jb_vals:
                s = sub[sub["coeff_jb"] == jb_v].sort_values(coeff_col)
                if len(s) < 2:
                    continue
                color = jb_colors.get(jb_v, "#999999")
                marker = jb_markers.get(jb_v, "o")
                ax.plot(s[coeff_col], s["refusal_rate_pct"], f"{marker}-",
                        color=color, linewidth=2, markersize=8,
                        markeredgecolor="white", markeredgewidth=1,
                        label=f"+ JB coeff = {jb_v}")
                for _, row in s.iterrows():
                    all_points.append((row[coeff_col], row["refusal_rate_pct"],
                                       f"{row['refusal_rate_pct']:.0f}%", color))
            _annotate_no_overlap(ax, all_points)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Refusal Rate (%)")
            ax.set_title(solo_cat.title() + " Sentiment Steering", fontweight="bold")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(100))
            ax.set_ylim(-5, 115)
            ax.legend(frameon=True, fontsize=8)
        fig.suptitle("Refusal Rate vs. Sentiment Coefficient",
                     fontweight="bold", fontsize=12, y=1.02)
        fig.tight_layout()
        _save_fig(fig, "refusal_vs_sentiment_coeff_combined", fig_dir)

    # ── 8b. Combined line plot: quality vs sentiment coefficient ──
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5.5), squeeze=False)
        fig.patch.set_facecolor("white")
        for idx, (jb_cat, solo_cat, coeff_col, xlabel) in enumerate(panels):
            ax = axes[0][idx]
            _style_ax(ax)
            all_points = []
            # Sentiment-only line
            solo = df[(df["category"] == solo_cat) & df[coeff_col].notna()].copy()
            if not solo.empty and len(solo) >= 2:
                solo = solo.sort_values(coeff_col)
                ax.plot(solo[coeff_col], solo["avg_quality"], f"{solo_marker}--",
                        color=solo_color, linewidth=2, markersize=8,
                        markeredgecolor="white", markeredgewidth=1,
                        label="Sentiment only (no JB)", alpha=0.8)
                for _, row in solo.iterrows():
                    all_points.append((row[coeff_col], row["avg_quality"],
                                       f"{row['avg_quality']:.1f}", solo_color))
            # Sentiment + JB lines
            sub = df[(df["category"] == jb_cat) & df[coeff_col].notna()].copy()
            jb_vals = sorted(sub["coeff_jb"].dropna().unique())
            for jb_v in jb_vals:
                s = sub[sub["coeff_jb"] == jb_v].sort_values(coeff_col)
                if len(s) < 2:
                    continue
                color = jb_colors.get(jb_v, "#999999")
                marker = jb_markers.get(jb_v, "o")
                ax.plot(s[coeff_col], s["avg_quality"], f"{marker}-",
                        color=color, linewidth=2, markersize=8,
                        markeredgecolor="white", markeredgewidth=1,
                        label=f"+ JB coeff = {jb_v}")
                for _, row in s.iterrows():
                    all_points.append((row[coeff_col], row["avg_quality"],
                                       f"{row['avg_quality']:.1f}", color))
            _annotate_no_overlap(ax, all_points)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Avg Quality (1–10)")
            ax.set_title(solo_cat.title() + " Sentiment Steering", fontweight="bold")
            ax.set_ylim(0, 11.5)
            ax.legend(frameon=True, fontsize=8)
        fig.suptitle("Response Quality vs. Sentiment Coefficient",
                     fontweight="bold", fontsize=12, y=1.02)
        fig.tight_layout()
        _save_fig(fig, "quality_vs_sentiment_coeff_combined", fig_dir)

    # ── 8c–8e. Combined line plots for SAE metrics vs sentiment coefficient ──
    sae_metrics = [
        ("delta_to_original_ratio", r"$\|\Delta\| \,/\, \|\mathbf{h}\|$",
         "Steering Magnitude vs. Sentiment Coefficient",
         "steering_magnitude_vs_sentiment_coeff_combined", None),
        ("reconstruction_error_l2", r"Reconstruction Error $\|\mathbf{h} - \mathrm{Dec}(\mathbf{z})\|_2$",
         "Reconstruction Error vs. Sentiment Coefficient",
         "recon_error_vs_sentiment_coeff_combined", None),
        ("recon_error_to_original_ratio", r"Recon Error / Original Norm",
         "Reconstruction Error Ratio vs. Sentiment Coefficient",
         "recon_error_ratio_vs_sentiment_coeff_combined", None),
        ("delta_l2", r"Steering Delta $\|\Delta\|_2$",
         "Steering Delta Norm vs. Sentiment Coefficient",
         "delta_norm_vs_sentiment_coeff_combined", None),
    ]
    for metric_col, ylabel, suptitle_text, save_name, ylim in sae_metrics:
        if metric_col not in df.columns or not panels:
            continue
        has_data = False
        for jb_cat, solo_cat, coeff_col, _ in panels:
            if not df[(df["category"].isin([jb_cat, solo_cat])) & df[coeff_col].notna() & df[metric_col].notna()].empty:
                has_data = True
                break
        if not has_data:
            continue
        fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5.5), squeeze=False)
        fig.patch.set_facecolor("white")
        for idx, (jb_cat, solo_cat, coeff_col, xlabel) in enumerate(panels):
            ax = axes[0][idx]
            _style_ax(ax)
            all_points = []
            fmt_fn = (lambda v: f"{v:.3f}") if df[metric_col].dropna().max() < 1 else (lambda v: f"{v:.0f}")
            # Sentiment-only line
            solo = df[(df["category"] == solo_cat) & df[coeff_col].notna() & df[metric_col].notna()].copy()
            if not solo.empty and len(solo) >= 2:
                solo = solo.sort_values(coeff_col)
                ax.plot(solo[coeff_col], solo[metric_col], f"{solo_marker}--",
                        color=solo_color, linewidth=2, markersize=8,
                        markeredgecolor="white", markeredgewidth=1,
                        label="Sentiment only (no JB)", alpha=0.8)
                for _, row in solo.iterrows():
                    all_points.append((row[coeff_col], row[metric_col],
                                       fmt_fn(row[metric_col]), solo_color))
            # Sentiment + JB lines
            sub = df[(df["category"] == jb_cat) & df[coeff_col].notna() & df[metric_col].notna()].copy()
            jb_vals = sorted(sub["coeff_jb"].dropna().unique())
            for jb_v in jb_vals:
                s = sub[sub["coeff_jb"] == jb_v].sort_values(coeff_col)
                if len(s) < 2:
                    continue
                color = jb_colors.get(jb_v, "#999999")
                marker = jb_markers.get(jb_v, "o")
                ax.plot(s[coeff_col], s[metric_col], f"{marker}-",
                        color=color, linewidth=2, markersize=8,
                        markeredgecolor="white", markeredgewidth=1,
                        label=f"+ JB coeff = {jb_v}")
                for _, row in s.iterrows():
                    all_points.append((row[coeff_col], row[metric_col],
                                       fmt_fn(row[metric_col]), color))
            _annotate_no_overlap(ax, all_points)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(solo_cat.title() + " Sentiment Steering", fontweight="bold")
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.legend(frameon=True, fontsize=8)
        fig.suptitle(suptitle_text, fontweight="bold", fontsize=12, y=1.02)
        fig.tight_layout()
        _save_fig(fig, save_name, fig_dir)

    print(f"\nAll figures saved to {fig_dir}/")


# ──────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────

def main():
    available_models = _discover_models()
    if not available_models:
        print("No steering result directories found. Exiting.")
        return
    print(f"Discovered models: {', '.join(available_models)}")

    current_model = available_models[0]

    while True:
        print(f"\nParsing results for model '{current_model}' …")
        entries = parse_all_results(current_model)
        print(f"Found {len(entries)} entries.\n")

        if not entries:
            print(f"No result files found for '{current_model}'. Skipping.")
            break

        # Launch labeler
        print("Opening labeling UI …")
        app = LabelerApp(entries, model=current_model, available_models=available_models)
        app.mainloop()

        # Check if user switched to another model
        next_model = getattr(app, "_next_model", None)

        # Aggregate current model
        print(f"\nAggregating results for '{current_model}' …")
        df = aggregate_results(entries, current_model)
        csv_out = _csv_output(current_model)
        df.to_csv(csv_out, index=False)
        print(f"Saved {csv_out}")
        print(df.to_string(index=False))

        # Graphs
        print(f"\nGenerating graphs for '{current_model}' …")
        generate_graphs(df, current_model)

        if next_model:
            current_model = next_model
            continue
        break

    print("\nDone!")


if __name__ == "__main__":
    main()
