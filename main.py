import tkinter as tk
from tkinter import ttk, filedialog, colorchooser
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import random
import math
import threading
from collections import deque
import json
import os

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

PREDEFINED_PALETTES = {
    "Biome": [
        (0,0,0), (28,82,106), (43,105,128), (57,128,149), (72,150,171), (86,173,192), (101,196,214), (115,219,235),
        (0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),
        (249,225,184), (244,215,165), (239,205,146), (234,195,127), (229,185,108), (224,175,89), (219,165,70), (214,155,51),
        (185,209,139), (170,198,121), (155,187,103), (140,176,85), (125,165,67), (110,154,49), (95,143,31), (80,132,13),
        (134,188,128), (119,173,113), (104,158,98), (89,143,83), (74,128,68), (59,113,53), (44,98,38), (29,83,23),
        (180,191,170), (171,181,162), (162,171,153), (153,161,145), (144,151,136), (135,141,128), (126,131,119), (117,121,111),
        (136,136,136), (150,150,150), (164,164,164), (178,178,178),
        (255,255,255), (245,245,245), (235,235,235), (225,225,225),
    ],
    "Default": [
        (0,0,0), (0,0,68), (0,17,102), (0,51,136), (0,85,170), (0,119,187),
        (0,153,221), (0,204,255), (34,221,255), (68,238,255), (102,255,255),
        (119,255,255), (136,255,255), (153,255,255), (170,255,255), (187,255,255),
        (0,68,0), (34,102,0), (34,136,0), (119,170,0), (187,221,0), (255,187,34),
        (238,170,34), (221,136,34), (204,136,34), (187,102,34), (170,85,34),
        (153,85,34), (136,68,34), (119,51,34), (85,51,17), (68,34,0), (255,255,255),
        (250,250,250), (245,245,245), (240,240,240), (235,235,235), (230,230,230),
        (225,225,225), (220,220,220), (215,215,215), (210,210,210), (205,205,205),
        (200,200,200), (195,195,195), (190,190,190), (185,185,185), (180,180,180),
        (175,175,175)
    ],
}

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
    def __init__(self, parent):
        super().__init__(parent)
        self.transient(parent)
        self.parent = parent
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
        self.parent.palette = list(self.current_palette)
        self.parent.recolor_map()
        self.destroy()

class FractalWorldGenerator:
    def __init__(self, params, palette, progress_callback=None):
        self.params = params
        self.x_range = params['width']
        self.y_range = params['height']
        self.seed = params['seed']
        self.progress_callback = progress_callback
        
        random.seed(self.seed)
        
        ice_seed = self.params.get('ice_seed', self.seed)
        moisture_seed = self.params.get('moisture_seed', self.seed)
        self.ice_noise = SimplexNoise(seed=ice_seed)
        self.moisture_noise = SimplexNoise(seed=moisture_seed)

        self.palette = palette
        self.INT_MIN_PLACEHOLDER = -2**31
        self.world_map = None
        self.color_map = None
        self.pil_image = None
        self.color_map_before_ice = None

    def _add_fault(self):
        flag1 = random.randint(0, 1)
        alpha, beta = (random.random() - 0.5) * np.pi, (random.random() - 0.5) * np.pi
        tan_b = np.tan(np.arccos(np.clip(np.cos(alpha) * np.cos(beta), -1.0, 1.0)))
        xsi = int(self.x_range / 2.0 - (self.x_range / np.pi) * beta)
        
        for phi in range(self.x_range):
            sin_val = self.sin_iter_phi[xsi - phi + self.x_range]
            theta = int((self.y_range_div_pi * np.arctan(sin_val * tan_b)) + self.y_range_div_2)
            theta = max(0, min(self.y_range - 1, theta))
            idx = (phi, theta)
            delta = 1 if flag1 else -1
            if self.world_map[idx] != self.INT_MIN_PLACEHOLDER:
                self.world_map[idx] += delta
            else:
                self.world_map[idx] = delta
                
    def _apply_erosion(self, erosion_passes):
        if erosion_passes <= 0: return
        
        heightmap = self.world_map.astype(np.float32)
        for i in range(erosion_passes):
            if self.progress_callback:
                progress = 50 + int(20 * (i / erosion_passes))
                self.progress_callback(progress, f"Eroding pass {i+1}/{erosion_passes}...")

            source_map = heightmap.copy()
            for y in range(self.y_range):
                for x in range(self.x_range):
                    current_height = source_map[x, y]
                    lowest_neighbor_height, lowest_nx, lowest_ny = current_height, x, y
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0: continue
                            nx, ny = (x + dx + self.x_range) % self.x_range, y + dy
                            if 0 <= ny < self.y_range and source_map[nx, ny] < lowest_neighbor_height:
                                lowest_neighbor_height, lowest_nx, lowest_ny = source_map[nx, ny], nx, ny
                    if lowest_neighbor_height < current_height:
                        sediment_amount = (current_height - lowest_neighbor_height) * 0.1
                        heightmap[x, y] -= sediment_amount
                        heightmap[lowest_nx, lowest_ny] += sediment_amount
        self.world_map = heightmap.astype(np.int32)

    def _apply_biomes(self, land_mask, land_min, land_max):
        biome_defs = {
            'rock':   {'base': 48, 'shades': 4}, 'tundra': {'base': 40, 'shades': 8},
            'desert': {'base': 16, 'shades': 8}, 'plains': {'base': 24, 'shades': 8},
            'forest': {'base': 32, 'shades': 8}
        }
        
        y_coords = np.arange(self.y_range)
        temperature = 1.0 - (np.abs(y_coords - self.y_range / 2.0) / (self.y_range / 2.0))
        temperature = np.tile(temperature, (self.x_range, 1))
        
        altitude = (self.world_map - land_min) / (land_max - land_min)
        
        moisture = np.zeros_like(self.world_map, dtype=np.float32)
        scale = 3.0
        for y in range(self.y_range):
            for x in range(self.x_range):
                angle = 2 * math.pi * (x / self.x_range)
                nx = (1 / (2*math.pi)) * scale * math.cos(angle)
                nz = (1 / (2*math.pi)) * scale * math.sin(angle)
                ny = y / (self.y_range/2) * scale
                moisture[x,y] = self.moisture_noise.fractal_noise3(nx, ny, nz, 5, 0.5, 2.0)
        moisture = (moisture - np.min(moisture)) / (np.max(moisture) - np.min(moisture))
        
        biome_names = ['rock', 'tundra', 'desert', 'plains', 'forest']
        conditions = [
            altitude > 0.75, temperature < 0.2, moisture < 0.33,
            moisture < 0.66, moisture >= 0.66
        ]
        
        biome_type_map = np.select(conditions, biome_names, default='plains')
        
        for name, props in biome_defs.items():
            current_biome_mask = (biome_type_map == name) & (land_mask)
            if not np.any(current_biome_mask): continue
            
            biome_altitudes = altitude[current_biome_mask]
            color_indices = props['base'] + np.floor(biome_altitudes * (props['shades'] - 0.01))
            self.color_map[current_biome_mask] = color_indices

    def _apply_ice_caps(self, percent_ice):
        if percent_ice <= 0: return

        noise_map = np.zeros((self.x_range, self.y_range))
        scale = 4.0
        for y in range(self.y_range):
            for x in range(self.x_range):
                angle = 2 * math.pi * (x / self.x_range)
                nx = (1 / (2*math.pi)) * scale * math.cos(angle)
                nz = (1 / (2*math.pi)) * scale * math.sin(angle)
                ny = y / (self.y_range/2) * scale
                noise_map[x, y] = self.ice_noise.fractal_noise3(nx, ny, nz, 6, 0.5, 2.0)
        noise_map = (noise_map - np.min(noise_map)) / (np.max(noise_map) - np.min(noise_map))

        min_alt, max_alt = np.min(self.world_map), np.max(self.world_map)
        if max_alt == min_alt: max_alt = min_alt + 1
        altitude_factor = (self.world_map - min_alt) / (max_alt - min_alt)
        y_coords = np.arange(self.y_range)
        latitude_factor_1d = np.abs(y_coords - (self.y_range - 1) / 2.0) / (self.y_range / 2.0)
        latitude_factor = np.tile(latitude_factor_1d, (self.x_range, 1))

        ice_score_map = (1.2 * latitude_factor) + (0.5 * altitude_factor) + (0.4 * noise_map)
        valid_pixels_mask = self.color_map > 0
        if not np.any(valid_pixels_mask): return
        ice_threshold = np.percentile(ice_score_map[valid_pixels_mask], 100 - percent_ice)
        ice_mask = (ice_score_map >= ice_threshold) & valid_pixels_mask
        if not np.any(ice_mask): return

        ice_altitudes = self.world_map[ice_mask]
        min_ice_alt, max_ice_alt = np.min(ice_altitudes), np.max(ice_altitudes)
        if max_ice_alt == min_ice_alt: max_ice_alt = min_ice_alt + 1
        
        ice_base_color_start = 53
        ice_num_base_colors = 3
        
        original_colors_under_ice = self.color_map[ice_mask]
        
        normalized_ice_alts = (ice_altitudes - min_ice_alt) / (max_ice_alt - min_ice_alt)
        base_ice_colors = ice_base_color_start + np.floor(normalized_ice_alts * (ice_num_base_colors - 0.01))
        
        terrain_modifier = (original_colors_under_ice % 2)
        final_ice_colors = base_ice_colors - terrain_modifier
        
        self.color_map[ice_mask] = final_ice_colors
    
    def generate(self):
        if self.progress_callback: self.progress_callback(0, "Generating faults...")
        self.world_map = np.full((self.x_range, self.y_range), self.INT_MIN_PLACEHOLDER, dtype=np.int32)
        self.y_range_div_2, self.y_range_div_pi = self.y_range / 2.0, self.y_range / np.pi
        self.sin_iter_phi = np.sin(np.arange(2 * self.x_range) * 2 * np.pi / self.x_range)
        
        num_faults = self.params['faults']
        for i in range(num_faults):
            self._add_fault()
            if self.progress_callback and i % 20 == 0:
                self.progress_callback(int(50 * (i / num_faults)), "Generating faults...")
        
        if self.progress_callback: self.progress_callback(50, "Calculating heightmap...")
        temp_map = self.world_map.copy()
        temp_map[temp_map == self.INT_MIN_PLACEHOLDER] = 0
        self.world_map = np.cumsum(temp_map, axis=1)
        
        self._apply_erosion(self.params.get('erosion'))
        
        if self.progress_callback: self.progress_callback(70, "Finalizing map...")
        self.finalize_map(
            self.params['water'], 
            self.params.get('ice', 15.0), 
            self.params.get('map_style', 'Biome')
        )
        
        if self.progress_callback: self.progress_callback(95, "Creating image...")
        self.pil_image = self.create_image()
        
        if self.progress_callback: self.progress_callback(100, "Done.")
        return self.pil_image

    def finalize_map(self, percent_water, percent_ice, map_style):
        min_z, max_z = np.min(self.world_map), np.max(self.world_map)
        if max_z == min_z: max_z = min_z + 1

        hist, bin_edges = np.histogram(self.world_map.flatten(), bins=256, range=(min_z, max_z))
        water_pixel_threshold = int((percent_water / 100.0) * (self.x_range * self.y_range))
        
        count, threshold_bin = 0, 0
        for i, num_pixels in enumerate(hist):
            count += num_pixels
            if count > water_pixel_threshold:
                threshold_bin = i; break
        water_level_threshold = bin_edges[threshold_bin]

        self.color_map = np.zeros_like(self.world_map, dtype=np.uint8)
        
        water_mask = self.world_map < water_level_threshold
        land_mask = ~water_mask
        
        water_min, water_max = min_z, water_level_threshold
        if water_max == water_min: water_max = water_min + 1
        self.color_map[water_mask] = 1 + np.floor(6.99 * (self.world_map[water_mask] - water_min) / (water_max - water_min))

        land_min, land_max = water_level_threshold, max_z
        if land_max == land_min: land_max = land_min + 1
        
        if map_style == 'Biome':
            self._apply_biomes(land_mask, land_min, land_max)
        else: # Terrain style
            normalized_altitude = (self.world_map[land_mask] - land_min) / (land_max - land_min)
            self.color_map[land_mask] = 16 + np.floor(15.99 * normalized_altitude)

        self.color_map_before_ice = self.color_map.copy()
        self._apply_ice_caps(percent_ice)

    def create_image(self):
        if self.color_map is None: return None
        
        clipped_color_map = np.clip(self.color_map, 0, len(self.palette) - 1)
        
        rgb_map = np.zeros((self.y_range, self.x_range, 3), dtype=np.uint8)
        for i, color in enumerate(self.palette):
            mask = clipped_color_map.T == i
            rgb_map[mask] = color
        return Image.fromarray(rgb_map, 'RGB')

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fractal World Generator")
        self.geometry("1100x800")
        
        self.pil_image, self.tk_image, self.generator = None, None, None
        
        self.zoom = 1.0
        self.view_offset = [0, 0]
        self.pan_start_pos = None
        
        self.placemarks = []
        self.adding_placemark = False

        self.main_frame = ttk.Frame(self, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.controls_frame = ttk.Labelframe(self.main_frame, text="Controls", padding="10")
        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.image_frame = ttk.Labelframe(self.main_frame, text="Generated Map", padding="10")
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.image_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.palette = list(PREDEFINED_PALETTES["Biome"])
        
        self.params = {
            'width': tk.IntVar(value=640), 'height': tk.IntVar(value=320),
            'seed': tk.IntVar(value=random.randint(0, 100000)),
            'ice_seed': tk.IntVar(value=random.randint(0, 100000)),
            'moisture_seed': tk.IntVar(value=random.randint(0, 100000)),
            'map_style': tk.StringVar(value='Biome'),
            'projection': tk.StringVar(value='Equirectangular'),
            'rotation_x': tk.DoubleVar(value=0.0),
            'rotation_y': tk.DoubleVar(value=0.0),
            'faults': tk.IntVar(value=200),
            'water': tk.DoubleVar(value=60.0),
            'ice': tk.DoubleVar(value=15.0),
            'erosion': tk.IntVar(value=5),
            'simulation_event': tk.StringVar(value='Ice Age Cycle'),
            'simulation_frames': tk.IntVar(value=20),
        }
        self._create_control_widgets()
        self.tooltip = MapTooltip(self)
        self.canvas.bind("<Configure>", self.redraw_canvas)
        self.canvas.bind("<MouseWheel>", self._on_zoom)
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<ButtonRelease-1>", self._on_pan_end)
        self.canvas.bind("<B1-Motion>", self._on_pan_move)
        self.canvas.bind("<Motion>", self._on_map_hover)
        self.canvas.bind("<Leave>", self._on_map_leave)
        
    def _create_control_widgets(self):
        row = 0
        self.controls_frame.grid_columnconfigure(1, weight=1)

        self._create_entry_widget("Width:", self.params['width'], row); row += 1
        self._create_entry_widget("Height:", self.params['height'], row); row += 1
        self._create_entry_widget("Seed:", self.params['seed'], row, include_random_button=True); row += 1
        self._create_entry_widget("Ice Seed:", self.params['ice_seed'], row, include_random_button=True); row += 1
        self._create_entry_widget("Moisture Seed:", self.params['moisture_seed'], row, include_random_button=True); row += 1
        
        ttk.Separator(self.controls_frame, orient='horizontal').grid(row=row, columnspan=3, sticky='ew', pady=10); row += 1
        
        view_frame = ttk.Labelframe(self.controls_frame, text="View Style")
        view_frame.grid(row=row, columnspan=3, sticky='ew', padx=5, pady=5); row += 1
        
        style_frame = ttk.Frame(view_frame); style_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(style_frame, text="Style:").pack(side=tk.LEFT)
        ttk.Radiobutton(style_frame, text="Biome", variable=self.params['map_style'], value="Biome", command=self.on_style_change).pack(side=tk.LEFT)
        ttk.Radiobutton(style_frame, text="Terrain", variable=self.params['map_style'], value="Terrain", command=self.on_style_change).pack(side=tk.LEFT)

        proj_frame = ttk.Frame(view_frame); proj_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(proj_frame, text="Projection:").pack(side=tk.LEFT)
        ttk.Radiobutton(proj_frame, text="2D", variable=self.params['projection'], value="Equirectangular", command=self.on_projection_change).pack(side=tk.LEFT)
        ttk.Radiobutton(proj_frame, text="Globe", variable=self.params['projection'], value="Orthographic", command=self.on_projection_change).pack(side=tk.LEFT)

        self.rotation_frame = ttk.Labelframe(self.controls_frame, text="Globe Rotation")
        self.rotation_frame.grid(row=row, columnspan=3, sticky='ew', padx=5, pady=5); row += 1
        self._create_slider_widget("Yaw:", self.params['rotation_y'], -180, 180, 0, master=self.rotation_frame)
        self._create_slider_widget("Pitch:", self.params['rotation_x'], -90, 90, 2, master=self.rotation_frame)
        
        self.params['rotation_y'].trace_add('write', lambda *_: self.redraw_canvas())
        self.params['rotation_x'].trace_add('write', lambda *_: self.redraw_canvas())

        self._create_slider_widget("Faults:", self.params['faults'], 1, 1000, row); row += 2
        self._create_slider_widget("Water %:", self.params['water'], 0, 100, row); row += 2
        self._create_slider_widget("Ice %:", self.params['ice'], 0, 100, row); row += 2
        self._create_slider_widget("Erosion:", self.params['erosion'], 0, 50, row); row += 2
        
        ttk.Label(self.controls_frame, text="Preset Palettes:").grid(row=row, column=0, columnspan=3, sticky='w', padx=5); row += 1
        self.palette_combobox = ttk.Combobox(self.controls_frame, values=list(PREDEFINED_PALETTES.keys()), state="readonly")
        self.palette_combobox.set("Biome")
        self.palette_combobox.grid(row=row, columnspan=3, sticky='ew', padx=5, pady=(0,10)); row += 1
        self.palette_combobox.bind("<<ComboboxSelected>>", self.apply_predefined_palette)

        sim_frame = ttk.Labelframe(self.controls_frame, text="Age Simulator")
        sim_frame.grid(row=row, columnspan=3, sticky='ew', padx=5, pady=5); row += 1
        sim_top_frame = ttk.Frame(sim_frame)
        sim_top_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(sim_top_frame, text="Event:").pack(side=tk.LEFT)
        event_combo = ttk.Combobox(sim_top_frame, textvariable=self.params['simulation_event'], values=['Ice Age Cycle'], state="readonly")
        event_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        sim_bottom_frame = ttk.Frame(sim_frame)
        sim_bottom_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(sim_bottom_frame, text="Frames:").pack(side=tk.LEFT)
        frames_entry = ttk.Entry(sim_bottom_frame, textvariable=self.params['simulation_frames'], width=5)
        frames_entry.pack(side=tk.LEFT, padx=5)
        self.run_sim_button = ttk.Button(sim_bottom_frame, text="Run Simulation", command=self.start_age_simulation, state=tk.DISABLED)
        self.run_sim_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Separator(self.controls_frame, orient='horizontal').grid(row=row, columnspan=3, sticky='ew', pady=10); row += 1
        
        self.progress = ttk.Progressbar(self.controls_frame, orient='horizontal', mode='determinate')
        self.progress.grid(row=row, columnspan=3, sticky='ew', pady=(5,0)); row += 1
        self.status_label = ttk.Label(self.controls_frame, text="Ready")
        self.status_label.grid(row=row, columnspan=3, sticky='ew', padx=5, pady=(0,5)); row += 1

        action_frame = ttk.Frame(self.controls_frame)
        action_frame.grid(row=row, columnspan=3, pady=5); row += 1
        self.generate_button = ttk.Button(action_frame, text="Generate", command=self.start_generation)
        self.generate_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.ice_button = ttk.Button(action_frame, text="Regenerate Ice", command=self.regenerate_ice, state=tk.DISABLED)
        self.ice_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        recolor_frame = ttk.Frame(self.controls_frame)
        recolor_frame.grid(row=row, columnspan=3, pady=5); row += 1
        self.recolor_button = ttk.Button(recolor_frame, text="Recolor Map", command=self.recolor_map, state=tk.DISABLED)
        self.recolor_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.palette_button = ttk.Button(recolor_frame, text="Edit Palette", command=self.open_palette_editor)
        self.palette_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        file_frame = ttk.Frame(self.controls_frame); file_frame.grid(row=row, columnspan=3, pady=5); row += 1
        ttk.Button(file_frame, text="Load Preset", command=self.load_preset).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(file_frame, text="Save Preset", command=self.save_preset).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        save_frame = ttk.Frame(self.controls_frame); save_frame.grid(row=row, columnspan=3, pady=5); row += 1
        self.save_button = ttk.Button(save_frame, text="Save Image As...", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(fill=tk.X, expand=True, padx=5)

        self.on_projection_change()

    def _create_entry_widget(self, label_text, var, row, include_random_button=False):
        ttk.Label(self.controls_frame, text=label_text).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        entry = ttk.Entry(self.controls_frame, textvariable=var, width=10)
        entry.grid(row=row, column=1, sticky='ew', padx=5)
        if include_random_button:
            button = ttk.Button(self.controls_frame, text="🎲", width=3, command=lambda v=var: v.set(random.randint(0, 100000)))
            button.grid(row=row, column=2, sticky='w')
        
    def _create_slider_widget(self, label_text, var, from_, to, row, master=None):
        if master is None: master = self.controls_frame
        ttk.Label(master, text=label_text).grid(row=row, column=0, columnspan=3, sticky='w', padx=5)
        is_double_var = isinstance(var, tk.DoubleVar)
        command_func = (lambda val, v=var: v.set(float(val))) if is_double_var else (lambda val, v=var: v.set(int(float(val))))
        ttk.Scale(master, from_=from_, to=to, orient='horizontal', variable=var, command=command_func).grid(row=row+1, column=0, columnspan=2, sticky='ew', padx=5)
        ttk.Entry(master, textvariable=var, width=7).grid(row=row+1, column=2, sticky='w', padx=5)

    def on_style_change(self):
        if not self.generator: return
        style = self.params['map_style'].get()
        if style == 'Biome':
            self.palette_combobox.set('Biome')
        else:
            self.palette_combobox.set('Default')
        self.apply_predefined_palette(None)

    def on_projection_change(self):
        if self.params['projection'].get() == 'Orthographic':
            self.rotation_frame.grid()
        else:
            self.rotation_frame.grid_remove()
        self.redraw_canvas()

    def _on_mouse_down(self, event):
        # This will be used for placemarks in a future update
        self._on_pan_start(event)
    
    def _on_map_hover(self, event):
        if self.generator is None or self.generator.color_map is None:
            self.tooltip.hide()
            return
            
        map_x, map_y = self.canvas_to_image_coords(event.x, event.y)
        
        if map_x is not None and 0 <= map_x < self.generator.x_range and 0 <= map_y < self.generator.y_range:
            final_color_index = self.generator.color_map.T[map_y, map_x]
            final_name = self.get_biome_name_from_index(final_color_index)
            
            display_text = final_name
            if final_name == "Ice" and self.generator.color_map_before_ice is not None:
                underlying_color_index = self.generator.color_map_before_ice.T[map_y, map_x]
                underlying_name = self.get_biome_name_from_index(underlying_color_index)
                if underlying_name and underlying_name != "Ice":
                    display_text = f"Ice (over {underlying_name})"

            if display_text:
                self.tooltip.show(display_text, event.x_root, event.y_root)
            else:
                self.tooltip.hide()
        else:
            self.tooltip.hide()

    def _on_map_leave(self, event):
        self.tooltip.hide()
        
    def get_biome_name_from_index(self, index):
        if self.params['map_style'].get() == 'Biome':
            if 1 <= index <= 7: return "Water"
            if 16 <= index <= 23: return "Desert"
            if 24 <= index <= 31: return "Grassland"
            if 32 <= index <= 39: return "Forest"
            if 40 <= index <= 47: return "Tundra"
            if 48 <= index <= 51: return "Rock"
            if 52 <= index <= 55: return "Ice"
        else:
            if 1 <= index <= 15: return "Water"
            if 16 <= index <= 31: return "Land"
            if 32 <= index <= 55: return "Ice"
        return None
        
    def set_ui_state(self, is_generating):
        state = tk.DISABLED if is_generating else tk.NORMAL
        for widget in [self.generate_button, self.ice_button, self.recolor_button, self.palette_button, self.save_button, self.palette_combobox, self.run_sim_button]:
            if widget != self.generate_button:
                widget.config(state=tk.NORMAL if not is_generating and self.generator else tk.DISABLED)
        self.generate_button.config(state=state)

    def start_generation(self):
        self.set_ui_state(is_generating=True)
        self.zoom = 1.0
        self.view_offset = [0, 0]
        self.placemarks = []
        params_dict = {key: var.get() for key, var in self.params.items()}
        thread = threading.Thread(target=self.run_generation_in_thread, args=(params_dict,), daemon=True)
        thread.start()
        
    def run_generation_in_thread(self, params_dict):
        self.palette = list(PREDEFINED_PALETTES[params_dict['map_style']])
        self.generator = FractalWorldGenerator(params_dict, self.palette, self.update_generation_progress)
        self.pil_image = self.generator.generate()
        self.after(0, self.finalize_generation)

    def finalize_generation(self):
        self.redraw_canvas()
        self.set_ui_state(is_generating=False)
        self.status_label.config(text="Ready")

    def update_generation_progress(self, value, text):
        self.after(0, lambda: (
            self.progress.config(value=value),
            self.status_label.config(text=text)
        ))

    def redraw_canvas(self, event=None):
        if not hasattr(self, 'generator') or not self.generator or not self.generator.pil_image: return
        self.tooltip.hide()
        
        image_to_draw = self.generator.pil_image.copy()
        projection = self.params['projection'].get()

        if projection == 'Orthographic':
            source_array = np.array(image_to_draw.convert('RGB'))
            canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if canvas_w <= 1 or canvas_h <= 1: return

            rot_y, rot_x = math.radians(self.params['rotation_y'].get()), math.radians(self.params['rotation_x'].get())
            cy, sy, cx, sx = math.cos(rot_y), math.sin(rot_y), math.cos(rot_x), math.sin(rot_x)
            
            canvas_coords_x, canvas_coords_y = np.meshgrid(np.arange(canvas_w), np.arange(canvas_h))
            radius = min(canvas_w, canvas_h) / 2
            ndc_x, ndc_y = (canvas_coords_x - canvas_w / 2) / radius, (canvas_coords_y - canvas_h / 2) / radius
            
            z2 = 1 - ndc_x**2 - ndc_y**2
            visible_mask = z2 >= 0
            z = np.sqrt(z2[visible_mask])
            x_ndc, y_ndc = ndc_x[visible_mask], ndc_y[visible_mask]
            
            y_r1 = y_ndc * cx - z * sx
            z_r1 = y_ndc * sx + z * cx
            x_r2 = x_ndc * cy - z_r1 * sy
            z_r2 = x_ndc * sy + z_r1 * cy

            lat, lon = np.arcsin(-y_r1), np.arctan2(x_r2, z_r2)
            
            src_x = np.clip(((lon / math.pi)*0.5 + 0.5) * self.generator.x_range, 0, self.generator.x_range - 1)
            src_y = np.clip(((-lat / math.pi) + 0.5) * self.generator.y_range, 0, self.generator.y_range - 1)
            
            new_img_array = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            new_img_array[canvas_coords_y[visible_mask], canvas_coords_x[visible_mask]] = source_array[src_y.astype(int), src_x.astype(int)]

            final_image = Image.fromarray(new_img_array)
        else:
            img_w, img_h = image_to_draw.size
            view_w, view_h = img_w / self.zoom, img_h / self.zoom
            
            x0 = self.view_offset[0]
            y0 = max(0, min(self.view_offset[1], img_h - view_h))
            self.view_offset[1] = y0

            x0_wrapped = x0 % img_w
            
            box1_x_end = min(img_w, x0_wrapped + view_w)
            box1 = (x0_wrapped, y0, box1_x_end, y0 + view_h)
            crop1 = image_to_draw.crop(box1)
            
            stitched_image = Image.new('RGB', (int(view_w), int(view_h)))
            stitched_image.paste(crop1, (0, 0))
            
            if x0_wrapped + view_w > img_w:
                remaining_w = (x0_wrapped + view_w) - img_w
                box2 = (0, y0, remaining_w, y0 + view_h)
                crop2 = image_to_draw.crop(box2)
                stitched_image.paste(crop2, (crop1.width, 0))
            
            final_image = stitched_image

        resized_image = final_image.resize((self.canvas.winfo_width(), self.canvas.winfo_height()), Image.Resampling.NEAREST)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def canvas_to_image_coords(self, canvas_x, canvas_y):
        if self.params['projection'].get() == 'Orthographic':
            if not self.generator: return None, None
            canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
            rot_y, rot_x = math.radians(self.params['rotation_y'].get()), math.radians(self.params['rotation_x'].get())
            cy, sy, cx, sx = math.cos(rot_y), math.sin(rot_y), math.cos(rot_x), math.sin(rot_x)
            radius = min(canvas_w, canvas_h) / 2
            ndc_x, ndc_y = (canvas_x - canvas_w / 2) / radius, (canvas_y - canvas_h / 2) / radius
            z2 = 1 - ndc_x**2 - ndc_y**2
            if z2 < 0: return None, None
            z = math.sqrt(z2)
            y_r1, z_r1 = y_ndc * cx - z * sx, y_ndc * sx + z * cx
            x_r2, z_r2 = ndc_x * cy - z_r1 * sy, ndc_x * sy + z_r1 * cy
            lat, lon = math.asin(-y_r1), math.atan2(x_r2, z_r2)
            map_x = int(((lon / math.pi) * 0.5 + 0.5) * self.generator.x_range)
            map_y = int(((-lat / math.pi) + 0.5) * self.generator.y_range)
            return map_x, map_y
            
        if not hasattr(self, 'generator') or not self.generator or not self.generator.pil_image: return -1, -1
        
        img_w, img_h = self.generator.pil_image.size
        percent_x, percent_y = canvas_x / self.canvas.winfo_width(), canvas_y / self.canvas.winfo_height()
        view_w, view_h = img_w / self.zoom, img_h / self.zoom
        img_x = self.view_offset[0] + (percent_x * view_w)
        img_y = self.view_offset[1] + (percent_y * view_h)
        return int(img_x), int(img_y)

    def _on_zoom(self, event):
        if not self.generator or not self.generator.pil_image: return
        if self.params['projection'].get() == 'Orthographic': return
        
        factor = 1.1 if event.delta > 0 else 0.9
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        self.zoom *= factor
        self.zoom = max(0.1, min(self.zoom, 10))
        new_img_x, new_img_y = self.canvas_to_image_coords(event.x, event.y)
        self.view_offset[0] += (img_x - new_img_x)
        self.view_offset[1] += (img_y - new_img_y)
        self.redraw_canvas()

    def _on_pan_start(self, event):
        if self.adding_placemark: return
        self.pan_start_pos = (event.x, event.y)
        self.canvas.config(cursor="fleur")

    def _on_pan_end(self, event):
        self.pan_start_pos = None
        self.canvas.config(cursor="")

    def _on_pan_move(self, event):
        if self.pan_start_pos is None: return
        
        dx, dy = event.x - self.pan_start_pos[0], event.y - self.pan_start_pos[1]
        
        if self.params['projection'].get() == 'Orthographic':
            self.params['rotation_y'].set(self.params['rotation_y'].get() + dx * 0.5)
            self.params['rotation_x'].set(max(-90, min(90, self.params['rotation_x'].get() + dy * 0.5)))
        else:
            self.view_offset[0] -= dx / self.zoom
            self.view_offset[1] -= dy / self.zoom
        
        self.pan_start_pos = (event.x, event.y)
        self.redraw_canvas()

    def save_image(self):
        if not self.generator or not self.generator.pil_image: return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")])
        if file_path:
            try:
                img_with_placemarks = self.generator.pil_image.copy()
                draw = ImageDraw.Draw(img_with_placemarks)
                try: font = ImageFont.truetype("arial.ttf", 12)
                except IOError: font = ImageFont.load_default()
                for pm in self.placemarks:
                    x, y = pm['x'], pm['y']
                    draw.ellipse((x-2, y-2, x+2, y+2), fill='red', outline='black')
                    draw.text((x+5, y-6), pm['name'], fill="white", font=font, stroke_width=1, stroke_fill="black")
                img_with_placemarks.save(file_path)

            except Exception as e: tk.messagebox.showerror("Save Error", f"Failed to save image:\n{e}")

    def save_preset(self):
        params_to_save = {key: var.get() for key, var in self.params.items() if key != 'placemark_name'}
        params_to_save['placemarks'] = self.placemarks
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Preset File", "*.json"), ("All Files", "*.*")], title="Save Preset As")
        if file_path:
            try:
                with open(file_path, 'w') as f: json.dump(params_to_save, f, indent=4)
            except Exception as e: tk.messagebox.showerror("Save Error", f"Failed to save preset:\n{e}")

    def load_preset(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON Preset File", "*.json"), ("All Files", "*.*")], title="Load Preset")
        if file_path:
            try:
                with open(file_path, 'r') as f: loaded_params = json.load(f)
                for key, var in self.params.items():
                    if key in loaded_params: var.set(loaded_params[key])
                
                self.placemarks = loaded_params.get('placemarks', [])

                self.on_projection_change()
                self.on_style_change()
            except Exception as e: tk.messagebox.showerror("Load Error", f"Failed to load preset:\n{e}")

    def open_palette_editor(self):
        PaletteEditor(self).grab_set()

    def regenerate_ice(self):
        if not self.generator: return
        self.set_ui_state(is_generating=True)
        thread = threading.Thread(target=self.run_ice_regeneration_in_thread, daemon=True)
        thread.start()

    def run_ice_regeneration_in_thread(self):
        self.update_generation_progress(0, "Finalizing map with new ice...")
        self.generator.params['ice_seed'] = self.params['ice_seed'].get()
        self.generator.ice_noise = SimplexNoise(seed=self.generator.params['ice_seed'])
        self.generator.finalize_map(
            self.params['water'].get(),
            self.params['ice'].get(),
            self.params['map_style'].get()
        )
        self.update_generation_progress(50, "Creating image...")
        self.generator.pil_image = self.generator.create_image()
        self.update_generation_progress(100, "Done.")
        self.after(0, self.finalize_generation)

    def recolor_map(self):
        if not self.generator or self.generator.color_map is None: return
        self.set_ui_state(is_generating=True)
        thread = threading.Thread(target=self.run_recolor_in_thread, daemon=True)
        thread.start()
        
    def run_recolor_in_thread(self):
        self.update_generation_progress(0, "Updating styles...")
        self.generator.palette = self.palette
        self.generator.finalize_map(
            self.params['water'].get(),
            self.params['ice'].get(),
            self.params['map_style'].get()
        )
        self.update_generation_progress(50, "Creating image...")
        self.generator.pil_image = self.generator.create_image()
        self.update_generation_progress(100, "Done.")
        self.after(0, self.finalize_generation)

    def apply_predefined_palette(self, event=None):
        if not self.generator: return
        palette_name = self.palette_combobox.get()
        self.palette = list(PREDEFINED_PALETTES[palette_name])
        self.recolor_map()
        
    def start_age_simulation(self):
        if not self.generator:
            tk.messagebox.showwarning("No Map", "Please generate a map first.")
            return

        save_dir = filedialog.askdirectory(title="Select Folder to Save Animation Frames")
        if not save_dir:
            return

        self.set_ui_state(is_generating=True)
        
        sim_params = {
            'event': self.params['simulation_event'].get(),
            'frames': self.params['simulation_frames'].get(),
            'save_dir': save_dir,
            'base_filename': os.path.basename(save_dir) 
        }

        thread = threading.Thread(target=self.run_age_simulation_in_thread, args=(sim_params,), daemon=True)
        thread.start()

    def run_age_simulation_in_thread(self, sim_params):
        event, frames, save_dir, base_filename = sim_params['event'], sim_params['frames'], sim_params['save_dir'], sim_params['base_filename']
        
        start_water = self.params['water'].get()
        start_ice = self.params['ice'].get()
        
        if event == 'Ice Age Cycle':
            peak_ice = 90.0
            peak_water = start_water - 10
            midpoint = frames / 2.0
            
            for i in range(frames):
                self.update_generation_progress(int((i + 1) / frames * 100), f"Rendering frame {i+1}/{frames}...")
                
                if i < midpoint:
                    progress = i / midpoint
                    current_ice = start_ice + (peak_ice - start_ice) * progress
                    current_water = start_water + (peak_water - start_water) * progress
                else:
                    progress = (i - midpoint) / midpoint
                    current_ice = peak_ice + (start_ice - peak_ice) * progress
                    current_water = peak_water + (start_water - peak_water) * progress

                self.generator.finalize_map(current_water, current_ice, self.params['map_style'].get())
                frame_image = self.generator.create_image()

                draw = ImageDraw.Draw(frame_image)
                try: font = ImageFont.truetype("arial.ttf", 12)
                except IOError: font = ImageFont.load_default()
                for pm in self.placemarks:
                    x, y = pm['x'], pm['y']
                    draw.ellipse((x-2, y-2, x+2, y+2), fill='red', outline='black')
                    draw.text((x+5, y-6), pm['name'], fill="white", font=font, stroke_width=1, stroke_fill="black")
                
                frame_image.save(os.path.join(save_dir, f"{base_filename}_{i+1:03d}.png"))

        self.generator.finalize_map(start_water, start_ice, self.params['map_style'].get())
        self.generator.pil_image = self.generator.create_image()
        
        self.after(0, self.finalize_generation)

if __name__ == "__main__":
    app = App()
    app.mainloop()
