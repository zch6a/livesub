"""A caption bar that floats over every window: committed text in white, the words still being
recognized dimmed, and the translation below. Drag it anywhere; double-click to quit.

tkinter only, so nothing to install. Thread-safe: other threads call `show_*`, and the window
picks the latest state up from a queue on its own thread.
"""

from __future__ import annotations

import queue
import sys


class Overlay:
    def __init__(self, font_size: int = 22, width_ratio: float = 0.7, opacity: float = 0.82, lines: int = 2):
        import tkinter as tk  # imported here: the rest of livesub works without a display

        self.tk = tk
        self.q: queue.Queue[tuple[str, str]] = queue.Queue()
        self.root = root = tk.Tk()
        root.title("livesub")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", opacity)
        except tk.TclError:  # pragma: no cover - some X11 window managers
            pass
        root.configure(bg="#111")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w = int(sw * width_ratio)
        h = int(font_size * 2.1 * (lines + 1)) + 24
        root.geometry(f"{w}x{h}+{(sw - w) // 2}+{sh - h - int(sh * 0.08)}")
        family = "Microsoft YaHei UI" if sys.platform == "win32" else "Noto Sans CJK SC"
        self.text = tk.Text(root, bg="#111", fg="#fff", bd=0, highlightthickness=0, wrap="word",
                            font=(family, font_size), height=lines, padx=16, pady=8, cursor="fleur")
        self.text.tag_configure("tentative", foreground="#9a9a9a")
        self.text.tag_configure("center", justify="center")
        self.text.pack(fill="both", expand=True)
        self.translation = tk.Label(root, bg="#111", fg="#ffd479", font=(family, int(font_size * 0.85)),
                                    wraplength=w - 32, justify="center")
        self.translation.pack(fill="x", padx=16, pady=(0, 8))
        for widget in (root, self.text, self.translation):
            widget.bind("<ButtonPress-1>", self._grab)
            widget.bind("<B1-Motion>", self._drag)
            widget.bind("<Double-Button-1>", lambda e: self.close())
        root.bind("<Escape>", lambda e: self.close())
        self._committed, self._tentative, self._translated = "", "", ""
        self._xy = (0, 0)
        self.closed = False
        root.after(50, self._poll)

    # Called from any thread.
    def show_text(self, committed: str, tentative: str = "") -> None:
        self.q.put(("text", f"{committed}\0{tentative}"))

    def show_translation(self, text: str) -> None:
        self.q.put(("translation", text))

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        self.closed = True
        self.root.after(0, self.root.destroy)

    # Window thread.
    def _poll(self) -> None:
        changed = False
        while True:
            try:
                kind, value = self.q.get_nowait()
            except queue.Empty:
                break
            changed = True
            if kind == "text":
                self._committed, self._tentative = value.split("\0", 1)
            else:
                self._translated = value
        if changed:
            self.text.configure(state="normal")
            self.text.delete("1.0", "end")
            self.text.insert("end", self._committed, ("center",))
            if self._tentative:
                self.text.insert("end", (" " if self._committed else "") + self._tentative, ("center", "tentative"))
            self.text.see("end")
            self.text.configure(state="disabled")
            self.translation.configure(text=self._translated)
        if not self.closed:
            self.root.after(50, self._poll)

    def _grab(self, e) -> None:
        self._xy = (e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y())

    def _drag(self, e) -> None:
        self.root.geometry(f"+{e.x_root - self._xy[0]}+{e.y_root - self._xy[1]}")
