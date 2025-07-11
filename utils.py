import tkinter as tk
from tkinter import ttk, filedialog, colorchooser
import random
import math
import json

def generate_fantasy_name(name_fragments, used_names, max_retries=10):
    """Generates a unique fantasy name using various patterns."""
    for _ in range(max_retries):
        pattern = random.random()
        name = ""
        if pattern < 0.1 and 'single' in name_fragments and name_fragments['single']:
            name = random.choice(name_fragments['single'])
        elif pattern < 0.3:
            name = f"{random.choice(name_fragments['prefixes'])}{random.choice(name_fragments['suffixes'])}"
        elif pattern < 0.6:
            name = f"{random.choice(name_fragments['prefixes'])}{random.choice(name_fragments.get('vowels', ['a']))}{random.choice(name_fragments['suffixes'])}"
        else:
            name = f"{random.choice(name_fragments['prefixes'])}{random.choice(name_fragments['suffixes'])}"

        name = name.replace('--', '-').replace('\'\'', '\'').strip('-')
        
        if name not in used_names:
            used_names.add(name)
            return name.capitalize()

    # Failsafe if a unique name isn't found
    return f"{random.choice(name_fragments['prefixes'])}{random.choice(name_fragments['suffixes'])}".capitalize()

class SimplexNoise:
    """
    A pure Python implementation of simplex noise by Stefan Gustavson.
    Adapted from various sources to be a self-contained class.
    """
    def __init__(self, seed=None):
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        p = list(range(256))
        random.Random(seed).shuffle(p)
        self.perm = p * 2

        self.F2 = 0.5 * (math.sqrt(3.0) - 1.0)
        self.G2 = (3.0 - math.sqrt(3.0)) / 6.0
        
        self.F3 = 1.0 / 3.0
        self.G3 = 1.0 / 6.0

        self.grad3 = [
            (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
            (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
            (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1)
        ]
        
    def _dot(self, grad, x, y):
        return grad[0] * x + grad[1] * y
    
    def _dot3(self, grad, x, y, z):
        return grad[0] * x + grad[1] * y + grad[2] * z

    def noise2(self, x, y):
        s = (x + y) * self.F2
        i, j = math.floor(x + s), math.floor(y + s)
        t = (i + j) * self.G2
        X0, Y0 = i - t, j - t
        x0, y0 = x - X0, y - Y0
        i1, j1 = (1, 0) if x0 > y0 else (0, 1)
        x1, y1 = x0 - i1 + self.G2, y0 - j1 + self.G2
        x2, y2 = x0 - 1.0 + 2.0 * self.G2, y0 - 1.0 + 2.0 * self.G2
        ii, jj = i & 255, j & 255
        gi0 = self.perm[ii + self.perm[jj]] % 12
        gi1 = self.perm[ii + i1 + self.perm[jj + j1]] % 12
        gi2 = self.perm[ii + 1 + self.perm[jj + 1]] % 12
        t0 = 0.5 - x0 * x0 - y0 * y0
        n0 = t0**4 * self._dot(self.grad3[gi0], x0, y0) if t0 > 0 else 0.0
        t1 = 0.5 - x1 * x1 - y1 * y1
        n1 = t1**4 * self._dot(self.grad3[gi1], x1, y1) if t1 > 0 else 0.0
        t2 = 0.5 - x2 * x2 - y2 * y2
        n2 = t2**4 * self._dot(self.grad3[gi2], x2, y2) if t2 > 0 else 0.0
        return 70.0 * (n0 + n1 + n2)

    def noise3(self, x, y, z):
        s = (x + y + z) * self.F3
        i, j, k = math.floor(x+s), math.floor(y+s), math.floor(z+s)
        t = (i + j + k) * self.G3
        X0, Y0, Z0 = i - t, j - t, k - t
        x0, y0, z0 = x - X0, y - Y0, z - Z0
        if x0 >= y0:
            if y0 >= z0: i1,j1,k1=1,0,0; i2,j2,k2=1,1,0
            elif x0 >= z0: i1,j1,k1=1,0,0; i2,j2,k2=1,0,1
            else: i1,j1,k1=0,0,1; i2,j2,k2=1,0,1
        else:
            if y0 < z0: i1,j1,k1=0,0,1; i2,j2,k2=0,1,1
            elif x0 < z0: i1,j1,k1=0,1,0; i2,j2,k2=0,1,1
            else: i1,j1,k1=0,1,0; i2,j2,k2=1,1,0
        x1,y1,z1 = x0-i1+self.G3, y0-j1+self.G3, z0-k1+self.G3
        x2,y2,z2 = x0-i2+2*self.G3, y0-j2+2*self.G3, z0-k2+2*self.G3
        x3,y3,z3 = x0-1+3*self.G3, y0-1+3*self.G3, z0-1+3*self.G3
        ii,jj,kk = i&255, j&255, k&255
        gi0 = self.perm[ii+self.perm[jj+self.perm[kk]]] % 12
        gi1 = self.perm[ii+i1+self.perm[jj+j1+self.perm[kk+k1]]] % 12
        gi2 = self.perm[ii+i2+self.perm[jj+j2+self.perm[kk+k2]]] % 12
        gi3 = self.perm[ii+1+self.perm[jj+1+self.perm[kk+1]]] % 12
        t0 = 0.6 - x0*x0 - y0*y0 - z0*z0
        n0 = t0**4 * self._dot3(self.grad3[gi0], x0, y0, z0) if t0 > 0 else 0.0
        t1 = 0.6 - x1*x1 - y1*y1 - z1*z1
        n1 = t1**4 * self._dot3(self.grad3[gi1], x1, y1, z1) if t1 > 0 else 0.0
        t2 = 0.6 - x2*x2 - y2*y2 - z2*z2
        n2 = t2**4 * self._dot3(self.grad3[gi2], x2, y2, z2) if t2 > 0 else 0.0
        t3 = 0.6 - x3*x3 - y3*y3 - z3*z3
        n3 = t3**4 * self._dot3(self.grad3[gi3], x3, y3, z3) if t3 > 0 else 0.0
        return 32.0 * (n0 + n1 + n2 + n3)

    def fractal_noise3(self, x, y, z, octaves, persistence, lacunarity):
        total, frequency, amplitude, max_value = 0.0, 1.0, 1.0, 0.0
        for _ in range(octaves):
            total += self.noise3(x*frequency, y*frequency, z*frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity
        return total / max_value

class MapTooltip(tk.Toplevel):
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
            x1, y1, x2, y2 = col * swatch_size, row * swatch_size, (col + 1) * swatch_size, (row + 1) * swatch_size
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
            except Exception as e: tk.messagebox.showerror("Save Error", f"Failed to save palette:\n{e}")

    def load_palette(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON Palette File", "*.json"), ("All Files", "*.*")], title="Load Palette")
        if file_path:
            try:
                with open(file_path, 'r') as f: loaded_data = json.load(f)
                if isinstance(loaded_data, list) and all(isinstance(c, list) and len(c) == 3 for c in loaded_data):
                    self.current_palette = [tuple(c) for c in loaded_data]
                    self.draw_palette()
                else: tk.messagebox.showwarning("Load Warning", "Invalid palette file format.")
            except Exception as e: tk.messagebox.showerror("Load Error", f"Failed to load palette:\n{e}")

    def apply_and_close(self):
        self.apply_and_close_callback(self.current_palette)
        self.destroy()