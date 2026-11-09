"""Run without arguments for the desktop app, or use --batch for a saved sweep."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback
import webbrowser

try:
    from engine import (DEFAULT_CONFIG_PATH, FILTER_NAMES, HERE, image_directory,
                        load_config, read_gray, run_experiment, save_json, select_images, validate_config)
    from PIL import Image, ImageTk
except ModuleNotFoundError as exc:
    raise SystemExit(f"Missing Python dependency: {exc.name}\nInstall with:\n  python -m pip install -r \"{Path(__file__).with_name('requirements.txt')}\"") from exc


def open_path(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(str(path))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class FilterLab:
    def __init__(self, root, cfg: dict):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.root = tk, ttk, root
        self.cfg = copy.deepcopy(cfg)
        self.variables = {}
        self.enabled = {}
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False
        self.close_when_done = False
        self.last_run = None
        self.manifest = None
        self.results = {}
        self.paths = []
        self.photo = None
        self.current_entry = None
        self.root.title("MAGFilo Filter Lab")
        self.root.geometry("1420x930")
        self.root.minsize(1000, 650)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 17, "bold"))
        style.configure("Help.TLabel", foreground="#52657b")

        outer = ttk.Frame(root, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="MAGFilo Filter Lab", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Compare independent filters at native resolution. xN = apply the same filter N times.", style="Help.TLabel").pack(anchor="w", pady=(2, 10))
        panes = ttk.Panedwindow(outer, orient="horizontal")
        panes.pack(fill="both", expand=True)
        left = ttk.Frame(panes, width=385)
        right = ttk.Frame(panes)
        panes.add(left, weight=0)
        panes.add(right, weight=1)

        scroll = tk.Canvas(left, width=375, highlightthickness=0)
        bar = ttk.Scrollbar(left, orient="vertical", command=scroll.yview)
        scroll.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        scroll.pack(side="left", fill="both", expand=True)
        settings = ttk.Frame(scroll, padding=(0, 0, 12, 10))
        window = scroll.create_window((0, 0), window=settings, anchor="nw")
        settings.bind("<Configure>", lambda event: scroll.configure(scrollregion=scroll.bbox("all")))
        scroll.bind("<Configure>", lambda event: scroll.itemconfigure(window, width=event.width))
        # Only the settings pane scrolls when its own descendants have focus.
        def wheel(event):
            if str(event.widget).startswith(str(settings)) or event.widget == scroll:
                scroll.yview_scroll(int(-event.delta / 120), "units")
        root.bind_all("<MouseWheel>", wheel, add="+")

        self._group(settings, "Images and output")
        self._path_field(settings, "input_dir", "Dataset root or image folder")
        self._path_field(settings, "output_dir", "Results folder")
        self._field(settings, "count", "Number of images", int)
        self._field(settings, "selection", "Selection", str, choices=("evenly", "first", "random"))
        self._field(settings, "seed", "Random seed", int)
        self._field(settings, "repeats", "Pass counts (comma separated)", "int_list")
        ttk.Label(settings, text="Examples: 1,2,3,5,10 or 1,5,10,20\nEvenly = spread across the sorted filenames.", style="Help.TLabel").pack(anchor="w", pady=(2, 6))
        buttons = ttk.Frame(settings)
        buttons.pack(fill="x", pady=5)
        self.refresh_button = ttk.Button(buttons, text="Refresh images", command=self.refresh)
        self.refresh_button.pack(side="left")
        ttk.Button(buttons, text="Load JSON", command=self.load_settings).pack(side="left", padx=4)
        ttk.Button(buttons, text="Save JSON", command=self.save_settings).pack(side="left")

        self._group(settings, "Filters (original is always included)")
        for name in ("median", "bilateral", "clahe", "solar"):
            self.enabled[name] = tk.BooleanVar(value=name in cfg["filters"])
            ttk.Checkbutton(settings, text=FILTER_NAMES[name], variable=self.enabled[name]).pack(anchor="w", pady=(6, 2))
            if name == "median":
                self._field(settings, "median.kernel_size", "Kernel size (odd)", int)
            elif name == "bilateral":
                self._field(settings, "bilateral.diameter", "Diameter (odd)", int)
                self._field(settings, "bilateral.sigma_color", "Sigma color (intensity)", float)
                self._field(settings, "bilateral.sigma_space", "Sigma space (pixels)", float)
            elif name == "clahe":
                self._field(settings, "clahe.clip_limit", "Clip limit", float)
                self._field(settings, "clahe.tile_grid", "Tile grid (columns, rows)", "int_list")
                self._field(settings, "clahe.strength", "Blend strength (0 to 1)", float)
            else:
                ttk.Label(settings, text="Solar disk correction + filament enhancement\nAdapted from your existing preprocessing notebook.", style="Help.TLabel").pack(anchor="w")
                self._field(settings, "solar.radial_strength", "Radial strength (0 to 1)", float)
                self._field(settings, "solar.radial_gain_limits", "Radial gain limits (low, high)", "float_list")
                self._field(settings, "solar.normalized_mean", "Target mean (0 to 255)", float)
                self._field(settings, "solar.normalized_std", "Target standard deviation", float)
                self._field(settings, "solar.morphology_strength", "Filament enhancement (0 to 1)", float)
                self._field(settings, "solar.disk_margin", "Disk margin", float)
                self._field(settings, "solar.disk_stats_radius", "Statistics radius", float)
                self._field(settings, "solar.salience_size", "Salience working size", int)
                self._field(settings, "solar.blackhat_kernels", "Black-hat kernels (3 odd sizes)", "int_list")
        self._group(settings, "Display and CPU")
        self._field(settings, "thumbnail_size", "Comparison tile size", int)
        self._field(settings, "opencv_threads", "CPU threads", int)

        chooser = ttk.Frame(right, padding=(12, 0, 0, 8))
        chooser.pack(fill="x")
        ttk.Label(chooser, text="Image:").pack(side="left")
        self.image_choice = ttk.Combobox(chooser, state="readonly", width=44)
        self.image_choice.pack(side="left", fill="x", expand=True, padx=6)
        self.image_choice.bind("<<ComboboxSelected>>", lambda event: self.show_selected())
        self.preview_button = ttk.Button(chooser, text="Preview selected", command=lambda: self.start(preview=True))
        self.preview_button.pack(side="left")
        self.batch_button = ttk.Button(chooser, text="Export all selected", command=self.start)
        self.batch_button.pack(side="left", padx=(6, 0))
        self.summary = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.summary, padding=(12, 0, 0, 6), style="Help.TLabel", wraplength=950).pack(anchor="w")
        display = ttk.Frame(right, padding=(12, 0, 0, 0))
        display.pack(fill="both", expand=True)
        display.rowconfigure(0, weight=1)
        display.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(display, background="#101820", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        horizontal = ttk.Scrollbar(display, orient="horizontal", command=self.canvas.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        vertical = ttk.Scrollbar(display, orient="vertical", command=self.canvas.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        self.canvas.bind("<Button-1>", self.open_tile)
        self.canvas.bind("<MouseWheel>", lambda event: self.canvas.yview_scroll(int(-event.delta / 120), "units"))
        ttk.Label(right, text="Click a comparison tile to inspect at 100% or 200% and toggle the original.", padding=(12, 6), style="Help.TLabel").pack(anchor="w")

        bottom = ttk.Frame(outer, padding=(0, 10, 0, 0))
        bottom.pack(fill="x")
        self.progress = ttk.Progressbar(bottom, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 5))
        self.status = tk.StringVar(value="Ready")
        ttk.Label(bottom, textvariable=self.status).pack(side="left", fill="x", expand=True)
        self.cancel_button = ttk.Button(bottom, text="Cancel", command=self.cancel_job, state="disabled")
        self.cancel_button.pack(side="right")
        self.gallery_button = ttk.Button(bottom, text="Open gallery", command=self.open_gallery, state="disabled")
        self.gallery_button.pack(side="right", padx=5)
        self.folder_button = ttk.Button(bottom, text="Open results folder", command=self.open_folder, state="disabled")
        self.folder_button.pack(side="right")
        self.refresh()
        root.after(100, self.poll)

    def _group(self, parent, title):
        self.ttk.Separator(parent).pack(fill="x", pady=(10, 6))
        self.ttk.Label(parent, text=title, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))

    def _value(self, key):
        value = self.cfg
        for part in key.split("."):
            value = value[part]
        return value

    def _field(self, parent, key, label, kind, choices=None):
        frame = self.ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        self.ttk.Label(frame, text=label).pack(side="left")
        value = self._value(key)
        var = self.tk.StringVar(value=", ".join(map(str, value)) if isinstance(value, list) else str(value))
        self.variables[key] = (var, kind)
        entry = self.ttk.Combobox(frame, textvariable=var, values=choices, state="readonly", width=15) if choices else self.ttk.Entry(frame, textvariable=var, width=17)
        entry.pack(side="right")

    def _path_field(self, parent, key, label):
        self.ttk.Label(parent, text=label).pack(anchor="w", pady=(4, 1))
        frame = self.ttk.Frame(parent)
        frame.pack(fill="x")
        var = self.tk.StringVar(value=self._value(key))
        self.variables[key] = (var, str)
        self.ttk.Entry(frame, textvariable=var).pack(side="left", fill="x", expand=True)
        self.ttk.Button(frame, text="...", width=3, command=lambda: self.browse(var)).pack(side="right", padx=(3, 0))

    def browse(self, var):
        from tkinter import filedialog
        folder = filedialog.askdirectory(parent=self.root, initialdir=var.get() if Path(var.get()).is_dir() else str(HERE))
        if folder:
            var.set(folder)

    def collect(self):
        cfg = copy.deepcopy(self.cfg)
        for key, (var, kind) in self.variables.items():
            raw = var.get().strip()
            try:
                if kind in ("int_list", "float_list"):
                    cast = int if kind == "int_list" else float
                    value = [cast(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip()]
                else:
                    value = kind(raw)
            except ValueError:
                raise ValueError(f"Invalid value for {key}: {raw!r}") from None
            target = cfg
            parts = key.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
        cfg["filters"] = ["none"] + [name for name, var in self.enabled.items() if var.get()]
        return validate_config(cfg)

    def error(self, message):
        from tkinter import messagebox
        self.status.set(str(message))
        messagebox.showerror("MAGFilo Filter Lab", str(message), parent=self.root)

    def refresh(self):
        try:
            self.cfg = self.collect()
            self.paths = select_images(self.cfg)
            self.results = {}
            self.image_choice["values"] = [f"{i:02d}. {p.name}" for i, p in enumerate(self.paths, 1)]
            self.image_choice.current(0)
            self.show_selected()
            self.status.set(f"Selected {len(self.paths)} images from {image_directory(self.cfg['input_dir'])}")
        except (ValueError, OSError) as exc:
            self.error(exc)

    def show_selected(self):
        index = self.image_choice.current()
        if not 0 <= index < len(self.paths):
            return
        source = self.paths[index]
        self.current_entry = self.results.get(str(source))
        try:
            if self.current_entry:
                with Image.open(self.current_entry["path"]) as saved:
                    image = saved.copy()
                self.summary.set(f"Saved comparison: {self.current_entry['path'].parent.parent.name}. Scroll to see all passes; click a tile for details.")
            else:
                image = Image.fromarray(read_gray(source))
                native = image.size
                image.thumbnail((780, 730), Image.Resampling.LANCZOS)
                self.summary.set(f"{source.name} | {native[0]} x {native[1]} | Original. Preview selected to compare filters.")
            self.photo = ImageTk.PhotoImage(image, master=self.root)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
            self.canvas.configure(scrollregion=(0, 0, image.width, image.height))
            self.canvas.xview_moveto(0)
            self.canvas.yview_moveto(0)
        except Exception as exc:
            self.error(exc)

    def load_settings(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(parent=self.root, initialdir=str(HERE), filetypes=[("JSON settings", "*.json")])
        if not path:
            return
        try:
            self.cfg = load_config(Path(path))
            for key, (var, _) in self.variables.items():
                value = self._value(key)
                var.set(", ".join(map(str, value)) if isinstance(value, list) else str(value))
            for name, var in self.enabled.items():
                var.set(name in self.cfg["filters"])
            self.refresh()
        except (ValueError, OSError) as exc:
            self.error(exc)

    def save_settings(self):
        from tkinter import filedialog
        try:
            cfg = self.collect()
            path = filedialog.asksaveasfilename(parent=self.root, initialdir=str(HERE), initialfile="my_settings.json", defaultextension=".json", filetypes=[("JSON settings", "*.json")])
            if path:
                save_json(Path(path), cfg)
                self.status.set(f"Saved settings: {path}")
        except (ValueError, OSError) as exc:
            self.error(exc)

    def set_busy(self, busy):
        self.busy = busy
        for button in (self.preview_button, self.batch_button, self.refresh_button):
            button.configure(state="disabled" if busy else "normal")
        self.cancel_button.configure(state="normal" if busy else "disabled")

    def start(self, preview=False):
        if self.busy:
            return
        try:
            cfg = self.collect()
            selected = select_images(cfg)
            current = self.paths[self.image_choice.current()] if self.paths and self.image_choice.current() >= 0 else None
            if preview:
                if current not in selected:
                    raise ValueError("Image selection changed. Click Refresh images, then choose an image to preview.")
                paths = [current]
            else:
                paths = selected
                self.paths = selected
                self.image_choice["values"] = [f"{i:02d}. {p.name}" for i, p in enumerate(selected, 1)]
                self.image_choice.current(0)
            self.cfg = cfg
            self.results = {}
            self.current_entry = None
            self.cancel.clear()
            self.progress["value"] = 0
            self.set_busy(True)
            self.status.set("Starting native-resolution filter sweep...")

            def worker():
                try:
                    def report(done, total, message):
                        self.events.put(("progress", done, total, message))
                    result = run_experiment(cfg, paths, report, self.cancel, "preview" if preview else "run")
                    self.events.put(("done", result))
                except Exception:
                    self.events.put(("error", traceback.format_exc()))
            threading.Thread(target=worker, name="filter-lab-worker", daemon=True).start()
        except (ValueError, OSError) as exc:
            self.error(exc)

    def cancel_job(self):
        self.cancel.set()
        self.status.set("Cancelling after the current filter pass; saving completed results...")
        self.cancel_button.configure(state="disabled")

    def poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "progress":
                    _, done, total, message = event
                    self.progress.configure(maximum=total, value=done)
                    self.status.set(message)
                elif event[0] == "done":
                    self.last_run, self.manifest = event[1]
                    self.set_busy(False)
                    for entry in self.manifest["comparisons"]:
                        self.results[entry["source"]] = {"path": self.last_run / entry["sheet"], "manifest": self.manifest, "run": self.last_run}
                    completed = sum(entry["complete"] for entry in self.manifest["comparisons"])
                    self.status.set(f"{self.manifest['status']}: {completed} image(s), {self.manifest['elapsed_seconds']:.1f}s. Saved: {self.last_run}")
                    for button in (self.gallery_button, self.folder_button):
                        button.configure(state="normal")
                    self.show_selected()
                    if self.close_when_done:
                        self.root.destroy()
                        return
                elif event[0] == "error":
                    self.set_busy(False)
                    if self.close_when_done:
                        self.root.destroy()
                        return
                    self.error(event[1])
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def open_tile(self, event):
        if not self.current_entry:
            return
        manifest = self.current_entry["manifest"]
        cfg = manifest["config"]
        size = cfg["thumbnail_size"]
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        if x < 10 or y < 50:
            return
        col = int((x - 10) // (size + 10))
        row = int((y - 50) // (size + 48))
        methods = [f for f in cfg["filters"] if f != "none"] or ["none"]
        columns = [0] + cfg["repeats"] if methods != ["none"] else [0]
        if row >= len(methods) or col >= len(columns):
            return
        source = str(self.paths[self.image_choice.current()])
        passes = columns[col]
        method = "none" if passes == 0 else methods[row]
        records = [r for r in manifest["records"] if r["source"] == source]
        rec = next((r for r in records if r["filter"] == method and r["passes"] == passes), None)
        original = next((r for r in records if r["filter"] == "none"), None)
        if rec and original:
            self.detail_view(self.current_entry["run"], rec, original)

    def detail_view(self, run_dir, record, original):
        window = self.tk.Toplevel(self.root)
        window.title(f"{Path(record['source']).name} | {FILTER_NAMES[record['filter']]} x{record['passes']}")
        window.geometry("1080x850")
        toolbar = self.ttk.Frame(window, padding=8)
        toolbar.pack(fill="x")
        zoom = self.tk.StringVar(value="Fit")
        combo = self.ttk.Combobox(toolbar, textvariable=zoom, values=("Fit", "50%", "100%", "200%"), state="readonly", width=8)
        combo.pack(side="left")
        show_original = self.tk.BooleanVar(value=False)
        mode = self.ttk.Checkbutton(toolbar, text="Show original (same zoom and position)", variable=show_original)
        mode.pack(side="left", padx=10)
        label = self.ttk.Label(toolbar, text="Drag to pan; use scrollbars to inspect details.")
        label.pack(side="left")
        if record["note"]:
            self.ttk.Label(window, text=record["note"], padding=8, foreground="#9b5700").pack(fill="x")
        display = self.ttk.Frame(window)
        display.pack(fill="both", expand=True)
        display.rowconfigure(0, weight=1)
        display.columnconfigure(0, weight=1)
        canvas = self.tk.Canvas(display, background="#101820", highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        sx = self.ttk.Scrollbar(display, orient="horizontal", command=canvas.xview)
        sy = self.ttk.Scrollbar(display, orient="vertical", command=canvas.yview)
        sx.grid(row=1, column=0, sticky="ew")
        sy.grid(row=0, column=1, sticky="ns")
        canvas.configure(xscrollcommand=sx.set, yscrollcommand=sy.set)
        with Image.open(run_dir / record["file"]) as file:
            filtered = file.copy()
        with Image.open(run_dir / original["file"]) as file:
            raw = file.copy()

        def render(event=None):
            image = raw if show_original.get() else filtered
            if zoom.get() == "Fit":
                factor = min(max(canvas.winfo_width() - 8, 100) / image.width, max(canvas.winfo_height() - 8, 100) / image.height, 1)
            else:
                factor = float(zoom.get().strip("%")) / 100
            resized = image.resize((max(1, round(image.width * factor)), max(1, round(image.height * factor))), Image.Resampling.NEAREST if factor >= 1 else Image.Resampling.LANCZOS)
            canvas.photo = ImageTk.PhotoImage(resized, master=window)
            canvas.delete("all")
            canvas.create_image(0, 0, image=canvas.photo, anchor="nw")
            canvas.configure(scrollregion=(0, 0, resized.width, resized.height))
        combo.bind("<<ComboboxSelected>>", render)
        mode.configure(command=render)
        canvas.bind("<ButtonPress-1>", lambda event: canvas.scan_mark(event.x, event.y))
        canvas.bind("<B1-Motion>", lambda event: canvas.scan_dragto(event.x, event.y, gain=1))
        canvas.bind("<Configure>", lambda event: render() if zoom.get() == "Fit" else None)
        window.after(100, render)

    def open_gallery(self):
        if self.last_run:
            webbrowser.open((self.last_run / "index.html").as_uri())

    def open_folder(self):
        if self.last_run:
            open_path(self.last_run)

    def close(self):
        if self.busy:
            self.close_when_done = True
            self.cancel_job()
        else:
            self.root.destroy()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Native-resolution MAGFilo filter comparison. No arguments opens the desktop app.")
    parser.add_argument("--batch", action="store_true", help="Export comparisons without opening the GUI")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="JSON settings file; relative paths are relative to this file")
    parser.add_argument("--input", type=Path, help="Dataset root, test folder, or test_images folder")
    parser.add_argument("--output", type=Path, help="Parent folder for a new output run")
    parser.add_argument("--count", type=int, help="Number of images")
    parser.add_argument("--selection", choices=("evenly", "first", "random"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--repeats", type=int, nargs="+", help="Successive pass counts, e.g. 1 2 3 5 10 20")
    parser.add_argument("--filters", nargs="+", choices=list(FILTER_NAMES), help="Filters to compare; original is always included")
    parser.add_argument("--open", action="store_true", help="Open the HTML gallery after a batch export")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        for argument, setting in (("input", "input_dir"), ("output", "output_dir")):
            value = getattr(args, argument)
            if value is not None:
                cfg[setting] = str(value.resolve())
        for key in ("count", "selection", "seed", "repeats", "filters"):
            value = getattr(args, key)
            if value is not None:
                cfg[key] = value
        cfg = validate_config(cfg)
        if args.batch:
            paths = select_images(cfg)
            print(f"Input: {image_directory(cfg['input_dir'])}\nImages: {len(paths)} | Filters: {', '.join(cfg['filters'])} | Passes: {cfg['repeats']}", flush=True)
            last_report = [0.0]
            def report(done, total, message):
                now = time.monotonic()
                if now - last_report[0] > 2 or done == total:
                    print(f"[{done}/{total}] {message}", flush=True)
                    last_report[0] = now
            run_dir, manifest = run_experiment(cfg, paths, report)
            completed = sum(entry["complete"] for entry in manifest["comparisons"])
            print(f"\n{manifest['status']}: {completed}/{len(paths)} images in {manifest['elapsed_seconds']:.1f}s\nGallery: {run_dir / 'index.html'}", flush=True)
            for failure in manifest["failures"]:
                print(f"ERROR: {failure}", file=sys.stderr)
            if args.open:
                webbrowser.open((run_dir / "index.html").as_uri())
            return 0 if manifest["status"] == "completed" else (130 if manifest["status"] == "cancelled" else 1)
        import tkinter as tk
        root = tk.Tk()
        FilterLab(root, cfg)
        root.mainloop()
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
