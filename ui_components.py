# beanz-y/fractal_world_generator/fractal_world_generator-28f75751b57dacf83432892d2293f1e3754a3ba6/ui_components.py

import tkinter as tk
from tkinter import ttk, filedialog, colorchooser
import json

class MapTooltip(tk.Toplevel):
    """A tooltip window that appears at the cursor's position."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.overrideredirect(True)
        self.withdraw()
        self.label = ttk.Label(self, text="", justify='left',
                               background="#ffffe0", relief='solid', borderwidth=1,
                               font=("tahoma", "8", "normal"))
        self.label.pack(ipadx=1)

    def show(self, text, x, y):
        self.label.config(text=text)
        self.geometry(f"+{x+15}+{y+10}")
        self.deiconify()

    def hide(self):
        self.withdraw()

class PaletteEditor(tk.Toplevel):
    """A dialog window for editing the color palette."""
    def __init__(self, parent, apply_and_close_callback):
        super().__init__(parent)
        self.transient(parent)
        self.parent = parent
        self.apply_and_close_callback = apply_and_close_callback
        self.title("Palette Editor")
        self.current_palette = list(self.parent.palette)
        
        self.main_frame = ttk.Frame(self, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.main_frame, width=480, height=180)
        self.canvas.pack(fill=tk.X, pady=5)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(self.button_frame, text="Load Palette", command=self.load_palette).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="Save Palette", command=self.save_palette).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="Apply and Close", command=self.apply_and_close).pack(side=tk.RIGHT, padx=5)
        
        self.draw_palette()

    def draw_palette(self):
        self.canvas.delete("all")
        swatch_size = 30
        for i, color_tuple in enumerate(self.current_palette):
            hex_color = f"#{color_tuple[0]:02x}{color_tuple[1]:02x}{color_tuple[2]:02x}"
            row, col = i // 16, i % 16
            x1, y1 = col * swatch_size, row * swatch_size
            x2, y2 = (col + 1) * swatch_size, (row + 1) * swatch_size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=hex_color, outline="white")
        self.canvas.create_text(5, 160, text="Click a square to change its color.", fill="grey", anchor="w")

    def _on_canvas_click(self, event):
        swatch_size = 30
        col, row = event.x // swatch_size, event.y // swatch_size
        index = row * 16 + col
        if 0 <= index < len(self.current_palette):
            current_color_hex = f"#{self.current_palette[index][0]:02x}{self.current_palette[index][1]:02x}{self.current_palette[index][2]:02x}"
            new_color = colorchooser.askcolor(color=current_color_hex, title="Select a new color")
            if new_color and new_color[0]:
                self.current_palette[index] = tuple(int(c) for c in new_color[0])
                self.draw_palette()

    def save_palette(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Palette File", "*.json"), ("All Files", "*.*")], title="Save Palette As")
        if file_path:
            try:
                with open(file_path, 'w') as f: json.dump(self.current_palette, f)
            except Exception as e: tk.messagebox.showerror("Save Error", f"Failed to save palette:\n{e}", parent=self)

    def load_palette(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON Palette File", "*.json"), ("All Files", "*.*")], title="Load Palette")
        if file_path:
            try:
                with open(file_path, 'r') as f: loaded_data = json.load(f)
                if isinstance(loaded_data, list) and all(isinstance(c, list) and len(c) == 3 for c in loaded_data):
                    self.current_palette = [tuple(c) for c in loaded_data]
                    self.draw_palette()
                else: tk.messagebox.showwarning("Load Warning", "Invalid palette file format.", parent=self)
            except Exception as e: tk.messagebox.showerror("Load Error", f"Failed to load palette:\n{e}", parent=self)

    def apply_and_close(self):
        self.apply_and_close_callback(self.current_palette)
        self.destroy()