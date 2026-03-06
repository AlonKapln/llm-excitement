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
LABELS_FILE = os.path.join(BASE_DIR, "labels.json")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
CSV_OUTPUT = os.path.join(BASE_DIR, "final_analysis.csv")

CATEGORIES = ["positive", "negative", "jailbreak", "positive_jailbreak", "negative_jailbreak"]


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


def parse_all_results():
    """Load every JSON file across all 5 subdirectories.

    Returns a list of dicts, one per (file, entry_index) pair:
        {category, filename, entry_idx, original_prompt, steered_response,
         config, coefficients, avg_metrics}
    """
    entries = []
    for category in CATEGORIES:
        cat_dir = os.path.join(RESULTS_DIR, category)
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

    def __init__(self, entries):
        super().__init__()
        self.title("Steering Response Labeler")
        self.geometry("1060x820")
        self.minsize(800, 650)
        self.configure(bg=self.BG)

        self.entries = entries
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

    @staticmethod
    def _load_labels():
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, "r") as f:
                return json.load(f)
        return {}

    def _save_labels(self):
        with open(LABELS_FILE, "w") as f:
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

        # ── Top bar: progress label + font size controls ─────
        top = tk.Frame(self, bg=self.BG)
        top.pack(fill="x", padx=PAD, pady=(12, 0))

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


# ──────────────────────────────────────────────────────────────
# 3. Aggregate results
# ──────────────────────────────────────────────────────────────

def aggregate_results(entries):
    """Combine labels + metrics into a per-file DataFrame."""
    labels = {}
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
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


def _save_fig(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  Saved figures/{name}.pdf and .png")


def generate_graphs(df):
    """Generate all publication-quality figures."""
    _setup_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── 1. Bar chart: refusal rate per config, grouped by category ──
    fig, ax = plt.subplots(figsize=(7, 4))
    x_labels = []
    refusal_values = []
    colors_list = []
    palette = {
        "positive": "#4c72b0", "negative": "#dd8452", "jailbreak": "#55a868",
        "positive_jailbreak": "#c44e52", "negative_jailbreak": "#8172b3",
    }
    for _, row in df.sort_values(["category", "config"]).iterrows():
        label = row["config"]
        x_labels.append(label)
        refusal_values.append(row["refusal_rate_pct"])
        colors_list.append(palette.get(row["category"], "#999999"))

    ax.bar(range(len(x_labels)), refusal_values, color=colors_list, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Refusal Rate (%)")
    ax.set_title("Refusal Rate by Steering Configuration")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(100))
    # Legend
    handles = [Patch(color=palette[c], label=c.replace("_", " ").title()) for c in CATEGORIES if c in set(df["category"])]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=7)
    fig.tight_layout()
    _save_fig(fig, "refusal_rate_bar")

    # ── 2. Heatmaps for two-coefficient combos ──
    for cat, coeff_x, coeff_y, xlabel, ylabel in [
        ("positive_jailbreak", "coeff_pos", "coeff_jb", "Positive Coeff", "Jailbreak Coeff"),
        ("negative_jailbreak", "coeff_neg", "coeff_jb", "Negative Coeff", "Jailbreak Coeff"),
    ]:
        sub = df[df["category"] == cat].copy()
        if sub.empty or coeff_x not in sub.columns or coeff_y not in sub.columns:
            continue
        pivot = sub.pivot_table(index=coeff_y, columns=coeff_x, values="refusal_rate_pct")
        pivot = pivot.sort_index(ascending=False)

        fig, ax = plt.subplots(figsize=(5, 3.5))
        im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Refusal Rate (%) — {cat.replace('_', ' ').title()}")
        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8,
                            color="white" if val > 50 else "black")
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Refusal Rate (%)")
        fig.tight_layout()
        _save_fig(fig, f"heatmap_{cat}")

    # ── 3. Bar chart: average quality rating per config ──
    df_q = df.dropna(subset=["avg_quality"])
    if not df_q.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        df_q_sorted = df_q.sort_values(["category", "config"])
        x_labels_q = df_q_sorted["config"].tolist()
        quality_vals = df_q_sorted["avg_quality"].tolist()
        colors_q = [palette.get(row["category"], "#999999") for _, row in df_q_sorted.iterrows()]

        ax.bar(range(len(x_labels_q)), quality_vals, color=colors_q, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(x_labels_q)))
        ax.set_xticklabels(x_labels_q, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Average Quality Rating (1–10)")
        ax.set_title("Response Quality by Steering Configuration")
        ax.set_ylim(0, 10.5)
        handles = [Patch(color=palette[c], label=c.replace("_", " ").title()) for c in CATEGORIES if c in set(df_q["category"])]
        ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=7)
        fig.tight_layout()
        _save_fig(fig, "quality_bar")

    # ── 4. Scatter: quality vs refusal rate ──
    df_scatter = df.dropna(subset=["avg_quality"])
    if not df_scatter.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        for cat in CATEGORIES:
            sub = df_scatter[df_scatter["category"] == cat]
            if sub.empty:
                continue
            ax.scatter(sub["refusal_rate_pct"], sub["avg_quality"],
                       label=cat.replace("_", " ").title(),
                       color=palette.get(cat, "#999999"), s=50, edgecolors="white", linewidth=0.5)
        ax.set_xlabel("Refusal Rate (%)")
        ax.set_ylabel("Average Quality Rating (1–10)")
        ax.set_title("Quality vs. Refusal Rate")
        ax.set_xlim(-5, 105)
        ax.set_ylim(0, 10.5)
        ax.legend(frameon=True, fontsize=7)
        fig.tight_layout()
        _save_fig(fig, "quality_vs_refusal_scatter")

    print(f"\nAll figures saved to {FIGURES_DIR}/")


# ──────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────

def main():
    print("Parsing all steering result files …")
    entries = parse_all_results()
    print(f"Found {len(entries)} total entries across {len(CATEGORIES)} categories.\n")

    if not entries:
        print("No result files found. Exiting.")
        return

    # Launch labeler
    print("Opening labeling UI …")
    app = LabelerApp(entries)
    app.mainloop()

    # Aggregate
    print("\nAggregating results …")
    df = aggregate_results(entries)
    df.to_csv(CSV_OUTPUT, index=False)
    print(f"Saved {CSV_OUTPUT}")
    print(df.to_string(index=False))

    # Graphs
    print("\nGenerating graphs …")
    generate_graphs(df)

    print("\nDone!")


if __name__ == "__main__":
    main()
