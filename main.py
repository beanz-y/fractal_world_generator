import tkinter as tk
from tkinter import ttk, filedialog, colorchooser
from PIL import Image, ImageTk
import numpy as np
import random
import math
import threading
from collections import deque
import json
import os

# --- [MODIFIED] Predefined Palettes Dictionary ---
# Corrected "Parchment & Ink" to use a single solid color for all water.
PREDEFINED_PALETTES = {
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
    "Parchment & Ink": [
        (0,0,0), (240,230,210), (240,230,210), (240,230,210), (240,230,210), (240,230,210),
        (240,230,210), (240,230,210), (240,230,210), (240,230,210), (240,230,210),
        (240,230,210), (240,230,210), (240,230,210), (240,230,210), (240,230,210), # <-- All water is now parchment color
        (225,215,195), (220,210,190), (215,205,185), (210,200,180), (200,190,170),
        (190,180,160), (180,170,150), (170,160,140), (160,150,130), (150,140,120),
        (140,130,110), (130,120,100), (120,110,90), (110,100,80), (100,90,70), (90,80,60),
        (200,210,220), (205,215,225), (210,220,230), (215,225,235), (220,230,240),
        (225,235,245), (230,240,250), (225,235,245), (220,230,240), (215,225,235),
        (210,220,230), (205,215,225), (200,210,220), (195,205,215), (190,200,210),
        (185,195,205)
    ],
    "Vintage Atlas": [
        (0,0,0), (180,210,220), (185,215,225), (190,220,230), (195,225,235), (200,230,240),
        (205,235,245), (210,240,250), (215,245,255), (210,240,250), (205,235,245),
        (200,230,240), (195,225,235), (190,220,230), (185,215,225), (180,210,220),
        (200,220,180), (205,225,185), (210,230,190), (215,235,195), (220,230,190),
        (225,225,185), (230,220,180), (225,215,175), (220,210,170), (215,205,165),
        (210,200,160), (200,190,155), (190,180,150), (180,170,145), (170,160,140), (160,150,135),
        (235,245,250), (230,240,245), (225,235,240), (220,230,235), (215,225,230),
        (210,220,225), (205,215,220), (200,210,215), (195,205,210), (190,200,205),
        (185,195,200), (180,190,195), (175,185,190), (170,180,185), (165,175,180),
        (160,170,175)
    ],
    "Arid / Desert": [
        (0,0,0), (20,85,90), (30,95,100), (40,105,110), (50,115,120), (60,125,130),
        (70,135,140), (80,145,150), (90,155,160), (100,165,170), (110,175,180),
        (120,185,190), (130,195,200), (140,205,210), (150,215,220), (160,225,230),
        (210,200,160), (205,190,150), (200,180,140), (195,170,130), (190,160,120),
        (185,150,110), (180,140,100), (175,130,90), (170,120,80), (165,110,75),
        (160,100,70), (155,90,65), (150,85,60), (145,80,55), (140,75,50), (135,70,45),
        (255,255,255), (250,250,250), (245,245,245), (240,240,240), (235,235,235),
        (230,230,230), (225,225,225), (220,220,220), (215,215,215), (210,210,210),
        (205,205,205), (200,200,200), (195,195,195), (190,190,190), (185,185,185),
        (180,180,180)
    ],
    "Volcanic / Scorched": [
        (0,0,0), (100,0,0), (110,5,0), (120,10,0), (130,15,0), (140,20,0), (150,25,0),
        (160,30,0), (170,35,0), (180,40,0), (200,50,0), (210,60,0), (220,70,0),
        (230,80,0), (240,90,0), (255,100,0), (10,10,10), (20,20,20), (30,30,30),
        (40,25,25), (50,30,30), (60,35,35), (80,40,40), (100,45,45), (120,50,50),
        (140,55,55), (160,60,60), (180,65,65), (200,70,70), (220,75,75),
        (240,80,80), (255,90,90), (200,200,200), (195,195,195), (190,190,190),
        (185,185,185), (180,180,180), (175,175,175), (170,170,170), (165,165,165),
        (160,160,160), (155,155,155), (150,150,150), (145,145,145), (140,140,140),
        (135,135,135), (130,130,130), (125,125,125)
    ],
    "Alien / Twilight": [
        (0,0,0), (50,0,90), (60,0,100), (70,10,110), (80,20,120), (90,30,130),
        (100,40,140), (110,50,150), (120,60,160), (130,70,170), (140,80,180),
        (150,90,190), (160,100,200), (170,110,210), (180,120,220), (190,130,230),
        (0,70,80), (10,80,90), (20,90,100), (30,100,110), (40,120,120),
        (50,140,130), (60,160,140), (80,180,150), (100,200,160), (120,210,150),
        (140,220,140), (160,230,130), (180,240,120), (200,250,110),
        (220,255,100), (240,255,90), (255,223,223), (255,218,218), (255,213,213),
        (255,208,208), (255,203,203), (255,198,198), (255,193,193), (255,188,188),
        (255,183,183), (255,178,178), (255,173,173), (255,168,168), (255,163,163),
        (255,158,158), (255,153,153), (255,148,148)
    ]
}

class PaletteEditor(tk.Toplevel):
    """
    A Toplevel window for creating, editing, saving, and loading color palettes.
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.transient(parent)
        self.parent = parent
        self.title("Palette Editor")

        # Copy the parent's palette to work with
        self.current_palette = list(self.parent.palette)

        # --- Layout ---
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
        """Draws the color swatches on the canvas."""
        self.canvas.delete("all")
        swatch_size = 30
        for i, color_tuple in enumerate(self.current_palette):
            hex_color = f"#{color_tuple[0]:02x}{color_tuple[1]:02x}{color_tuple[2]:02x}"
            
            row = i // 16
            col = i % 16
            
            x1 = col * swatch_size
            y1 = row * swatch_size
            x2 = x1 + swatch_size
            y2 = y1 + swatch_size
            
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=hex_color, outline="white", tags=f"swatch_{i}")
        self.canvas.create_text(5, 160, text="Click a square to change its color.", fill="grey", anchor="w")


    def _on_canvas_click(self, event):
        """Handles clicks on the canvas to select and change a color."""
        swatch_size = 30
        col = event.x // swatch_size
        row = event.y // swatch_size
        index = row * 16 + col
        
        if 0 <= index < len(self.current_palette):
            current_color_hex = f"#{self.current_palette[index][0]:02x}{self.current_palette[index][1]:02x}{self.current_palette[index][2]:02x}"
            new_color = colorchooser.askcolor(color=current_color_hex, title="Select a new color")
            
            if new_color and new_color[0]:
                self.current_palette[index] = tuple(int(c) for c in new_color[0])
                self.draw_palette()

    def save_palette(self):
        """Saves the current palette to a JSON file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Palette File", "*.json"), ("All Files", "*.*")],
            title="Save Palette As"
        )
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.current_palette, f)
            except Exception as e:
                tk.messagebox.showerror("Save Error", f"Failed to save palette:\n{e}")

    def load_palette(self):
        """Loads a palette from a JSON file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON Palette File", "*.json"), ("All Files", "*.*")],
            title="Load Palette"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, list) and all(isinstance(c, list) and len(c) == 3 for c in loaded_data):
                        self.current_palette = [tuple(c) for c in loaded_data]
                        self.draw_palette()
                    else:
                        tk.messagebox.showwarning("Load Warning", "Invalid palette file format.")
            except Exception as e:
                tk.messagebox.showerror("Load Error", f"Failed to load palette:\n{e}")

    def apply_and_close(self):
        """Applies the new palette to the main app and closes the editor."""
        self.parent.palette = list(self.current_palette)
        self.parent.recolor_map()
        self.destroy()

class FractalWorldGenerator:
    def __init__(self, params, palette, progress_callback=None):
        self.x_range = params['width']
        self.y_range = params['height']
        self.num_faults = params['faults']
        self.percent_water = params['water']
        self.percent_ice = params['ice']
        self.seed = params['seed']
        self.erosion_passes = params.get('erosion', 0)
        self.progress_callback = progress_callback
        
        random.seed(self.seed)
        
        self.palette = palette
        self.INT_MIN_PLACEHOLDER = -2**31
        self.world_map = None
        self.color_map = None

    def _add_fault(self):
        flag1 = random.randint(0, 1)
        alpha = (random.random() - 0.5) * np.pi
        beta = (random.random() - 0.5) * np.pi
        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        cos_alpha_cos_beta = np.clip(cos_alpha * cos_beta, -1.0, 1.0)
        tan_b = np.tan(np.arccos(cos_alpha_cos_beta))
        xsi = int(self.x_range / 2.0 - (self.x_range / np.pi) * beta)

        for phi in range(self.x_range // 2):
            sin_val = self.sin_iter_phi[xsi - phi + self.x_range]
            theta = int((self.y_range_div_pi * np.arctan(sin_val * tan_b)) + self.y_range_div_2)
            theta = max(0, min(self.y_range - 1, theta))
            idx = (phi, theta)
            current_val = self.world_map[idx]
            
            if flag1: delta = 1
            else: delta = -1
            
            if current_val != self.INT_MIN_PLACEHOLDER: self.world_map[idx] += delta
            else: self.world_map[idx] = delta
                
    def _apply_erosion(self):
        if self.erosion_passes <= 0: return

        heightmap = self.world_map.astype(np.float32)

        for _ in range(self.erosion_passes):
            source_map = heightmap.copy()
            for y in range(self.y_range):
                for x in range(self.x_range):
                    current_height = source_map[x, y]
                    lowest_neighbor_height = current_height
                    lowest_nx, lowest_ny = x, y

                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0: continue
                            nx, ny = (x + dx + self.x_range) % self.x_range, y + dy
                            if 0 <= ny < self.y_range:
                                neighbor_height = source_map[nx, ny]
                                if neighbor_height < lowest_neighbor_height:
                                    lowest_neighbor_height = neighbor_height
                                    lowest_nx, lowest_ny = nx, ny

                    if lowest_neighbor_height < current_height:
                        sediment_amount = (current_height - lowest_neighbor_height) * 0.1
                        heightmap[x, y] -= sediment_amount
                        heightmap[lowest_nx, lowest_ny] += sediment_amount
        
        self.world_map = heightmap.astype(np.int32)
                
    def _flood_fill(self, start_x, start_y, old_color, max_fill_percent):
        max_pixels_to_fill = (self.x_range * self.y_range) * (max_fill_percent / 100.0)
        if self.color_map[start_x, start_y] != old_color: return 0
        filled_pixels = 0
        q = deque([(start_x, start_y)])

        while q:
            x, y = q.popleft()
            if self.color_map[x, y] != old_color: continue

            if self.color_map[x, y] < 16: self.color_map[x, y] = 32
            else: self.color_map[x, y] += 17

            filled_pixels += 1
            if filled_pixels >= max_pixels_to_fill: return filled_pixels

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = (x + dx + self.x_range) % self.x_range, y + dy
                if 0 <= ny < self.y_range and self.color_map[nx, ny] == old_color: q.append((nx, ny))
        return filled_pixels

    def generate(self):
        self.world_map = np.full((self.x_range, self.y_range), self.INT_MIN_PLACEHOLDER, dtype=np.int32)
        self.y_range_div_2 = self.y_range / 2.0
        self.y_range_div_pi = self.y_range / np.pi
        self.sin_iter_phi = np.sin(np.arange(2 * self.x_range) * 2 * np.pi / self.x_range)
        
        for i in range(self.num_faults):
            self._add_fault()
            if self.progress_callback: self.progress_callback(i + 1)

        for j in range(self.x_range // 2):
            for i in range(1, self.y_range):
                self.world_map[j + self.x_range // 2, self.y_range - 1 - i] = self.world_map[j, i]

        temp_map = self.world_map.copy()
        temp_map[temp_map == self.INT_MIN_PLACEHOLDER] = 0
        self.world_map = np.cumsum(temp_map, axis=1)
        
        self._apply_erosion()
        self.finalize_map()
        return self.create_image()

    def finalize_map(self):
        min_z, max_z = np.min(self.world_map), np.max(self.world_map)
        if max_z == min_z: max_z = min_z + 1

        hist, bin_edges = np.histogram(self.world_map.flatten(), bins=256, range=(min_z, max_z))
        water_pixel_threshold = int((self.percent_water / 100.0) * (self.x_range * self.y_range))
        
        count = 0
        threshold_bin = 0
        for i, num_pixels in enumerate(hist):
            count += num_pixels
            if count > water_pixel_threshold:
                threshold_bin = i
                break
        
        water_level_threshold = bin_edges[threshold_bin]

        self.color_map = np.zeros_like(self.world_map, dtype=np.uint8)
        water_mask = self.world_map < water_level_threshold
        land_mask = ~water_mask
        
        water_min, water_max = min_z, water_level_threshold
        if water_max == water_min: water_max = water_min + 1
        self.color_map[water_mask] = 1 + np.floor(14.99 * (self.world_map[water_mask] - water_min) / (water_max - water_min))

        land_min, land_max = water_level_threshold, max_z
        if land_max == land_min: land_max = land_min + 1
        self.color_map[land_mask] = 16 + np.floor(15.99 * (self.world_map[land_mask] - land_min) / (land_max - land_min))

        self.color_map = np.clip(self.color_map, 1, 31)

        if self.percent_ice > 0:
            for x in range(self.x_range):
                old_color = self.color_map[x, 0]
                if old_color < 32: self._flood_fill(x, 0, old_color, self.percent_ice / 2.0)
            
            for x in range(self.x_range):
                old_color = self.color_map[x, self.y_range - 1]
                if old_color < 32: self._flood_fill(x, self.y_range - 1, old_color, self.percent_ice / 2.0)

    def create_image(self):
        if self.color_map is None: return None
        rgb_map = np.zeros((self.y_range, self.x_range, 3), dtype=np.uint8)
        for i, color in enumerate(self.palette):
            mask = self.color_map.T == i
            rgb_map[mask] = color
        return Image.fromarray(rgb_map, 'RGB')

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fractal World Generator")
        self.geometry("1100x800")
        
        self.pil_image = None
        self.tk_image = None
        self.generator = None
        
        self.main_frame = ttk.Frame(self, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.controls_frame = ttk.Labelframe(self.main_frame, text="Controls", padding="10")
        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.image_frame = ttk.Labelframe(self.main_frame, text="Generated Map", padding="10")
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.image_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.update_canvas())
        
        self.palette = list(PREDEFINED_PALETTES["Default"])

        self.params = {
            'width': tk.IntVar(value=640), 'height': tk.IntVar(value=320),
            'seed': tk.IntVar(value=random.randint(0, 100000)),
            'faults': tk.IntVar(value=200), 'water': tk.DoubleVar(value=60.0),
            'ice': tk.DoubleVar(value=10.0), 'erosion': tk.IntVar(value=5)
        }
        
        self._create_control_widgets()
        
    def _create_control_widgets(self):
        row = 0
        self.controls_frame.grid_columnconfigure(1, weight=1)

        self._create_entry_widget("Width:", self.params['width'], row)
        row += 1
        self._create_entry_widget("Height:", self.params['height'], row)
        row += 1
        self._create_entry_widget("Seed:", self.params['seed'], row, include_random_button=True)
        row += 1

        ttk.Separator(self.controls_frame, orient='horizontal').grid(row=row, columnspan=3, sticky='ew', pady=10)
        row += 1

        self._create_slider_widget("Faults:", self.params['faults'], 1, 1000, row); row += 2
        self._create_slider_widget("Water %:", self.params['water'], 0, 100, row); row += 2
        self._create_slider_widget("Ice %:", self.params['ice'], 0, 100, row); row += 2
        self._create_slider_widget("Erosion:", self.params['erosion'], 0, 50, row); row += 2
        
        ttk.Label(self.controls_frame, text="Preset Palettes:").grid(row=row, column=0, columnspan=3, sticky='w', padx=5)
        row += 1
        self.palette_combobox = ttk.Combobox(
            self.controls_frame,
            values=list(PREDEFINED_PALETTES.keys()),
            state="readonly"
        )
        self.palette_combobox.set("Default")
        self.palette_combobox.grid(row=row, columnspan=3, sticky='ew', padx=5, pady=(0,10))
        self.palette_combobox.bind("<<ComboboxSelected>>", self.apply_predefined_palette)
        row += 1
        
        ttk.Separator(self.controls_frame, orient='horizontal').grid(row=row, columnspan=3, sticky='ew', pady=10)
        row += 1
        
        self.progress = ttk.Progressbar(self.controls_frame, orient='horizontal', mode='determinate')
        self.progress.grid(row=row, columnspan=3, sticky='ew', pady=5)
        row += 1

        action_frame = ttk.Frame(self.controls_frame)
        action_frame.grid(row=row, columnspan=3, pady=5)
        self.generate_button = ttk.Button(action_frame, text="Generate", command=self.start_generation)
        self.generate_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.recolor_button = ttk.Button(action_frame, text="Recolor Map", command=self.recolor_map, state=tk.DISABLED)
        self.recolor_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        row += 1
        
        file_frame = ttk.Frame(self.controls_frame)
        file_frame.grid(row=row, columnspan=3, pady=5)
        self.palette_button = ttk.Button(file_frame, text="Edit Palette", command=self.open_palette_editor)
        self.palette_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.save_button = ttk.Button(file_frame, text="Save Image As...", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        row += 1
        
        preset_frame = ttk.Frame(self.controls_frame)
        preset_frame.grid(row=row, columnspan=3, pady=5)
        ttk.Button(preset_frame, text="Load Preset", command=self.load_preset).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(preset_frame, text="Save Preset", command=self.save_preset).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    def _create_entry_widget(self, label_text, var, row, include_random_button=False):
        ttk.Label(self.controls_frame, text=label_text).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        entry = ttk.Entry(self.controls_frame, textvariable=var, width=10)
        entry.grid(row=row, column=1, sticky='ew', padx=5)
        if include_random_button:
            button = ttk.Button(self.controls_frame, text="🎲", width=3, command=lambda: var.set(random.randint(0, 100000)))
            button.grid(row=row, column=2, sticky='w')
        
    def _create_slider_widget(self, label_text, var, from_, to, row):
        ttk.Label(self.controls_frame, text=label_text).grid(row=row, column=0, columnspan=3, sticky='w', padx=5)
        slider = ttk.Scale(self.controls_frame, from_=from_, to=to, orient='horizontal', variable=var,
                           command=lambda val, v=var: v.set(int(float(val))))
        slider.grid(row=row+1, column=0, columnspan=2, sticky='ew', padx=5)
        entry = ttk.Entry(self.controls_frame, textvariable=var, width=7)
        entry.grid(row=row+1, column=2, sticky='w', padx=5)

    def set_ui_state(self, is_generating):
        state = tk.DISABLED if is_generating else tk.NORMAL
        self.generate_button.config(state=state)
        self.palette_button.config(state=state)
        self.palette_combobox.config(state=state)

        save_state = tk.NORMAL if not is_generating and self.pil_image else tk.DISABLED
        self.save_button.config(state=save_state)
        self.recolor_button.config(state=save_state)

    def start_generation(self):
        self.set_ui_state(is_generating=True)
        params_dict = {key: var.get() for key, var in self.params.items()}
        self.progress.config(maximum=params_dict['faults'])
        self.progress['value'] = 0
        thread = threading.Thread(target=self.run_generation_in_thread, args=(params_dict,), daemon=True)
        thread.start()
        
    def run_generation_in_thread(self, params_dict):
        self.generator = FractalWorldGenerator(params_dict, self.palette, progress_callback=self.update_progress)
        self.pil_image = self.generator.generate()
        self.after(0, self.update_canvas)

    def update_progress(self, value):
        self.after(0, self.progress.config, {'value': value})

    def update_canvas(self):
        if self.pil_image:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                self.after(100, self.update_canvas)
                return

            img_w, img_h = self.pil_image.size
            scale = min(canvas_width / img_w, canvas_height / img_h)
            new_size = (int(img_w * scale), int(img_h * scale))

            resized_image = self.pil_image.resize(new_size, Image.Resampling.NEAREST)
            self.tk_image = ImageTk.PhotoImage(resized_image)
            
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width / 2, canvas_height / 2, anchor=tk.CENTER, image=self.tk_image)
        
        self.set_ui_state(is_generating=False)

    def save_image(self):
        if not self.pil_image: return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                self.pil_image.save(file_path)
            except Exception as e:
                tk.messagebox.showerror("Save Error", f"Failed to save image:\n{e}")

    def save_preset(self):
        params_dict = {key: var.get() for key, var in self.params.items()}
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Preset File", "*.json"), ("All Files", "*.*")],
            title="Save Preset As"
        )
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(params_dict, f, indent=4)
            except Exception as e:
                tk.messagebox.showerror("Save Error", f"Failed to save preset:\n{e}")

    def load_preset(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON Preset File", "*.json"), ("All Files", "*.*")],
            title="Load Preset"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    loaded_params = json.load(f)
                for key, var in self.params.items():
                    if key in loaded_params: var.set(loaded_params[key])
            except Exception as e:
                tk.messagebox.showerror("Load Error", f"Failed to load preset:\n{e}")

    def open_palette_editor(self):
        editor = PaletteEditor(self)
        editor.grab_set()

    def recolor_map(self):
        if not self.generator or self.generator.color_map is None: return
        self.generator.palette = self.palette
        self.pil_image = self.generator.create_image()
        self.update_canvas()

    def apply_predefined_palette(self, event=None):
        """Applies a palette from the PREDEFINED_PALETTES dictionary."""
        palette_name = self.palette_combobox.get()
        # Make a copy so we don't modify the original
        self.palette = list(PREDEFINED_PALETTES[palette_name])
        self.recolor_map()

if __name__ == "__main__":
    app = App()
    app.mainloop()