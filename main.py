import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFont # type: ignore
import numpy as np # type: ignore
import random
import math
import threading
from collections import deque
import json
import os
import time

from world_generator import FractalWorldGenerator, PREDEFINED_PALETTES, BIOME_DEFINITIONS
from utils import MapTooltip, PaletteEditor

# Add this line to disable the Decompression Bomb warning for very large images
Image.MAX_IMAGE_PIXELS = None

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
        self.adding_placemark = tk.BooleanVar(value=False)
        
        self.simulation_frames = []
        self.current_frame_index = -1

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
            'thaw_ice_seed': tk.IntVar(value=random.randint(0, 100000)),
            'map_style': tk.StringVar(value='Biome'),
            'projection': tk.StringVar(value='Equirectangular'),
            'rotation_x': tk.DoubleVar(value=0.0),
            'rotation_y': tk.DoubleVar(value=0.0),
            'faults': tk.IntVar(value=200),
            'water': tk.DoubleVar(value=60.0),
            'ice': tk.DoubleVar(value=15.0),
            'erosion': tk.IntVar(value=5),
            'altitude_temp_effect': tk.DoubleVar(value=0.5),
            'wind_direction': tk.StringVar(value='West to East'), # New parameter for wind
            'simulation_event': tk.StringVar(value='Ice Age Cycle'),
            'simulation_frames': tk.IntVar(value=20),
            'hex_grid_visible': tk.BooleanVar(value=False),
            'hex_grid_size': tk.IntVar(value=50),
        }
        self._create_control_widgets()
        self.tooltip = MapTooltip(self)
        self.canvas.bind("<Configure>", self.redraw_canvas)
        self.canvas.bind("<MouseWheel>", self._on_zoom)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<ButtonRelease-1>", self._on_pan_end)
        self.canvas.bind("<B1-Motion>", self._on_pan_move)
        self.canvas.bind("<Motion>", self._on_map_hover)
        self.canvas.bind("<Leave>", self._on_map_leave)
        
    def _create_control_widgets(self):
        row = 0
        self.controls_frame.grid_columnconfigure(1, weight=1)

        # --- Generation Parameters ---
        gen_frame = ttk.Labelframe(self.controls_frame, text="Generation")
        gen_frame.grid(row=row, columnspan=3, sticky='ew', padx=5, pady=5); row += 1
        gen_frame.grid_columnconfigure(1, weight=1)
        
        g_row = 0
        self._create_entry_widget("Width:", self.params['width'], g_row, master=gen_frame); g_row += 1
        self._create_entry_widget("Height:", self.params['height'], g_row, master=gen_frame); g_row += 1
        self._create_entry_widget("Seed:", self.params['seed'], g_row, include_random_button=True, master=gen_frame); g_row += 1
        self._create_entry_widget("Ice Seed:", self.params['ice_seed'], g_row, include_random_button=True, master=gen_frame); g_row += 1
        self._create_slider_widget("Faults:", self.params['faults'], 1, 2000, g_row, master=gen_frame); g_row += 1
        self._create_slider_widget("Erosion:", self.params['erosion'], 0, 50, g_row, master=gen_frame); g_row += 1
        
        # --- Climate Parameters ---
        climate_frame = ttk.Labelframe(self.controls_frame, text="Climate")
        climate_frame.grid(row=row, columnspan=3, sticky='ew', padx=5, pady=5); row += 1
        climate_frame.grid_columnconfigure(1, weight=1)
        
        c_row = 0
        self._create_slider_widget("Water %:", self.params['water'], 0, 100, c_row, master=climate_frame); c_row += 1
        self._create_slider_widget("Ice %:", self.params['ice'], 0, 100, c_row, master=climate_frame); c_row += 1
        self._create_slider_widget("Altitude Temp. Effect:", self.params['altitude_temp_effect'], 0, 1, c_row, master=climate_frame); c_row += 1

        ttk.Label(climate_frame, text="Wind Direction:").grid(row=c_row, column=0, columnspan=2, sticky='w', padx=5); c_row += 1
        wind_combo = ttk.Combobox(climate_frame, textvariable=self.params['wind_direction'], 
                                  values=['West to East', 'East to West', 'North to South', 'South to North'], state="readonly")
        wind_combo.grid(row=c_row, column=0, columnspan=2, sticky='ew', padx=5, pady=(0,5)); c_row += 1

        # --- View and Display ---
        view_frame = ttk.Labelframe(self.controls_frame, text="Display")
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
        self._create_slider_widget("Pitch:", self.params['rotation_x'], -90, 90, 1, master=self.rotation_frame)
        
        self.params['rotation_y'].trace_add('write', lambda *_: self.redraw_canvas())
        self.params['rotation_x'].trace_add('write', lambda *_: self.redraw_canvas())
        
        ttk.Label(self.controls_frame, text="Preset Palettes:").grid(row=row, column=0, columnspan=3, sticky='w', padx=5); row += 1
        self.palette_combobox = ttk.Combobox(self.controls_frame, values=list(PREDEFINED_PALETTES.keys()), state="readonly")
        self.palette_combobox.set("Biome")
        self.palette_combobox.grid(row=row, columnspan=3, sticky='ew', padx=5, pady=(0,10)); row += 1
        self.palette_combobox.bind("<<ComboboxSelected>>", self.apply_predefined_palette)
        
        # ... (Rest of the controls: Sim, Overlay, Placemark, etc.)
        # This part is unchanged, so I'll omit it for brevity.
        # Just ensure the new frames above are placed correctly.
        
        sim_frame = ttk.Labelframe(self.controls_frame, text="Age Simulator")
        sim_frame.grid(row=row, columnspan=3, sticky='ew', padx=5, pady=5); row += 1
        sim_top_frame = ttk.Frame(sim_frame)
        sim_top_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(sim_top_frame, text="Event:").pack(side=tk.LEFT)
        event_combo = ttk.Combobox(sim_top_frame, textvariable=self.params['simulation_event'], values=['Ice Age Cycle'], state="readonly")
        event_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        sim_middle_frame = ttk.Frame(sim_frame)
        sim_middle_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(sim_middle_frame, text="Thaw Ice Seed:").pack(side=tk.LEFT, padx=(5,0))
        thaw_entry = ttk.Entry(sim_middle_frame, textvariable=self.params['thaw_ice_seed'], width=10)
        thaw_entry.pack(side=tk.LEFT, padx=5)
        thaw_random_button = ttk.Button(sim_middle_frame, text="🎲", width=3, command=lambda: self.params['thaw_ice_seed'].set(random.randint(0, 100000)))
        thaw_random_button.pack(side=tk.LEFT)
        
        sim_bottom_frame = ttk.Frame(sim_frame)
        sim_bottom_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(sim_bottom_frame, text="Frames:").pack(side=tk.LEFT)
        frames_entry = ttk.Entry(sim_bottom_frame, textvariable=self.params['simulation_frames'], width=5)
        frames_entry.pack(side=tk.LEFT, padx=5)
        self.run_sim_button = ttk.Button(sim_bottom_frame, text="Run Simulation", command=self.start_age_simulation, state=tk.DISABLED)
        self.run_sim_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        overlay_frame = ttk.Labelframe(self.controls_frame, text="Overlay Tools")
        overlay_frame.grid(row=row, columnspan=3, sticky='ew', padx=5, pady=5); row += 1
        overlay_frame.grid_columnconfigure(0, weight=1)
        ttk.Checkbutton(overlay_frame, text="Show Hex Grid", variable=self.params['hex_grid_visible'], command=self.redraw_canvas).grid(row=0, column=0, sticky='w', padx=5)
        self._create_slider_widget("Hex Size:", self.params['hex_grid_size'], 10, 200, 1, master=overlay_frame)
        self.params['hex_grid_size'].trace_add('write', lambda *_: self.redraw_canvas())
        
        placemark_frame = ttk.Labelframe(self.controls_frame, text="Placemark Tools")
        placemark_frame.grid(row=row, columnspan=3, sticky='ew', padx=5, pady=5); row += 1
        add_placemark_button = ttk.Checkbutton(placemark_frame, text="Add Placemark", variable=self.adding_placemark, command=self.toggle_placemark_mode)
        add_placemark_button.pack(side=tk.LEFT, padx=5)
        
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

        self.frame_viewer_row = row; row += 1
        self.frame_viewer_frame = ttk.Labelframe(self.controls_frame, text="Frame Viewer")
        
        frame_nav_frame = ttk.Frame(self.frame_viewer_frame)
        frame_nav_frame.pack(pady=5)
        self.prev_frame_button = ttk.Button(frame_nav_frame, text="< Prev", command=self.show_previous_frame, state=tk.DISABLED)
        self.prev_frame_button.pack(side=tk.LEFT, padx=5)
        self.frame_label = ttk.Label(frame_nav_frame, text="0 / 0")
        self.frame_label.pack(side=tk.LEFT, padx=5)
        self.next_frame_button = ttk.Button(frame_nav_frame, text="Next >", command=self.show_next_frame, state=tk.DISABLED)
        self.next_frame_button.pack(side=tk.LEFT, padx=5)
        self.on_projection_change()
        # ... Rest of the file is identical to your latest version ...
    def _create_entry_widget(self, label_text, var, row, include_random_button=False, master=None):
        if master is None: master = self.controls_frame
        ttk.Label(master, text=label_text).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        entry = ttk.Entry(master, textvariable=var, width=10)
        entry.grid(row=row, column=1, sticky='ew', padx=5)
        if include_random_button:
            button = ttk.Button(master, text="🎲", width=3, command=lambda v=var: v.set(random.randint(0, 100000)))
            button.grid(row=row, column=2, sticky='w')
        
    def _create_slider_widget(self, label_text, var, from_, to, row, master=None):
        if master is None: master = self.controls_frame
        
        container = ttk.Frame(master)
        
        if label_text:
            ttk.Label(container, text=label_text).grid(row=0, column=0, sticky='w', padx=5)
        
        is_double_var = isinstance(var, tk.DoubleVar)
        command_func = (lambda val, v=var: v.set(float(val))) if is_double_var else (lambda val, v=var: v.set(int(float(val))))
        
        scale = ttk.Scale(container, from_=from_, to=to, orient='horizontal', variable=var, command=command_func)
        scale.grid(row=1, column=0, sticky='ew', padx=5)
        
        entry = ttk.Entry(container, textvariable=var, width=7)
        entry.grid(row=1, column=1, sticky='w', padx=5)
        
        container.grid(row=row, column=0, columnspan=2, sticky='ew')
        container.grid_columnconfigure(0, weight=1)

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
            self.redraw_canvas()
        else:
            self.rotation_frame.grid_remove()
            self.redraw_canvas()

    def _on_canvas_press(self, event):
        if self.adding_placemark.get():
            self._add_placemark_at_click(event)
        else:
            self._on_pan_start(event)
    
    def _add_placemark_at_click(self, event):
        if not self.generator: return
        
        placemark_name = simpledialog.askstring("New Placemark", "Enter a name for this location:", parent=self)
        if not placemark_name:
            return
            
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        if img_x is not None:
            self.placemarks.append({'name': placemark_name, 'x': img_x, 'y': img_y})
            self.redraw_canvas()

    def _on_map_hover(self, event):
        if self.generator is None or self.generator.color_map is None:
            self.tooltip.hide()
            return
            
        map_x, map_y = self.canvas_to_image_coords(event.x, event.y)
        
        if map_x is not None and 0 <= map_x < self.generator.x_range and 0 <= map_y < self.generator.y_range:
            elevation = self.generator.world_map.T[map_y, map_x]
            final_color_index = self.generator.color_map.T[map_y, map_x]
            final_name = self.get_biome_name_from_index(final_color_index)
            
            display_text = final_name
            if final_name and final_name.lower() == "polar ice" and self.generator.color_map_before_ice is not None:
                underlying_color_index = self.generator.color_map_before_ice.T[map_y, map_x]
                underlying_name = self.get_biome_name_from_index(underlying_color_index)
                if underlying_name and underlying_name.lower() != "polar ice":
                    display_text = f"Polar Ice (over {underlying_name})"

            if display_text:
                full_display_text = f"{display_text}\nCoords: ({map_x}, {map_y})\nElevation: {elevation}"
                self.tooltip.show(full_display_text, event.x_root, event.y_root)
            else:
                self.tooltip.hide()
        else:
            self.tooltip.hide()

    def _on_map_leave(self, event):
        self.tooltip.hide()
        
    def get_biome_name_from_index(self, index):
        if self.params['map_style'].get() == 'Biome':
            if 53 <= index <= 55:
                return "Polar Ice"
            for name, props in BIOME_DEFINITIONS.items():
                if props['idx'] <= index < props['idx'] + props.get('shades', 1):
                    return name.replace('_', ' ').title()
        
        if 1 <= index <= 7: return "Water"
        if 16 <= index <= 31: return "Land"
        return None
        
    def set_ui_state(self, is_generating):
        state = tk.DISABLED if is_generating else tk.NORMAL
        for widget in [self.generate_button, self.ice_button, self.recolor_button, self.palette_button, self.save_button, self.palette_combobox, self.run_sim_button]:
            if widget != self.generate_button:
                widget.config(state=tk.NORMAL if not is_generating and self.generator else tk.DISABLED)
        self.generate_button.config(state=state)

    def start_generation(self):
        self.set_ui_state(is_generating=True)
        self.frame_viewer_frame.grid_remove()
        self.simulation_frames = []
        self.current_frame_index = -1
        self.zoom = 1.0
        self.view_offset = [0, 0]
        self.placemarks = []
        params_dict = {key: var.get() for key, var in self.params.items()}
        thread = threading.Thread(target=self.run_generation_in_thread, args=(params_dict,), daemon=True)
        thread.start()
        
    def run_generation_in_thread(self, params_dict):
        map_style = params_dict.get('map_style', 'Biome')
        palette_name = 'Biome' if map_style == 'Biome' else 'Default'
        self.palette = list(PREDEFINED_PALETTES[palette_name])

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
        
    def _draw_placemarks_on_image(self, image):
        if not self.placemarks:
            return image
        
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()
            
        for pm in self.placemarks:
            x, y = pm['x'], pm['y']
            draw.ellipse((x-3, y-3, x+3, y+3), fill='red', outline='black')
            draw.text((x+5, y-8), pm['name'], fill="black", font=font, stroke_width=2, stroke_fill="white")
            
        return image
        
    def _draw_hex_grid_on_image(self, image):
        if not self.params['hex_grid_visible'].get():
            return image

        size = self.params['hex_grid_size'].get()
        if size <= 0: return image

        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        hex_width = math.sqrt(3) * size
        vert_dist = size * 1.5
        
        q_end = int(image.width / hex_width) + 2
        r_end = int(image.height / vert_dist) + 2
        
        for r in range(-1, r_end):
            for q in range(-1, q_end):
                hex_center_x = q * hex_width
                if r % 2 != 0:
                    hex_center_x += hex_width / 2.0
                hex_center_y = r * vert_dist
                
                points = []
                for i in range(6):
                    angle_deg = 60 * i + 30
                    angle_rad = math.pi / 180 * angle_deg
                    point_x = hex_center_x + size * math.cos(angle_rad)
                    point_y = hex_center_y + size * math.sin(angle_rad)
                    points.append((point_x, point_y))
                
                draw.polygon(points, outline=(0, 0, 0, 128), fill=None)
                
        return Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')


    def redraw_canvas(self, event=None):
        if not hasattr(self, 'generator') or not self.generator or not self.generator.pil_image: return
        self.tooltip.hide()
        
        image_with_overlays = self._draw_hex_grid_on_image(
            self._draw_placemarks_on_image(self.generator.pil_image.copy())
        )
        projection = self.params['projection'].get()

        if projection == 'Orthographic':
            source_array = np.array(image_with_overlays.convert('RGB'))
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
            x_ndc_masked, y_ndc_masked = ndc_x[visible_mask], ndc_y[visible_mask]
            
            y_r1 = y_ndc_masked * cx - z * sx
            z_r1 = y_ndc_masked * sx + z * cx
            x_r2 = x_ndc_masked * cy - z_r1 * sy
            z_r2 = x_ndc_masked * sy + z_r1 * cy

            lat, lon = np.arcsin(-y_r1), np.arctan2(x_r2, z_r2)
            
            src_x = np.clip(((lon / math.pi)*0.5 + 0.5) * self.generator.x_range, 0, self.generator.x_range - 1)
            src_y = np.clip(((-lat / math.pi) + 0.5) * self.generator.y_range, 0, self.generator.y_range - 1)
            
            new_img_array = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            new_img_array[canvas_coords_y[visible_mask], canvas_coords_x[visible_mask]] = source_array[src_y.astype(int), src_x.astype(int)]

            final_image = Image.fromarray(new_img_array)
        else:
            img_w, img_h = image_with_overlays.size
            view_w, view_h = img_w / self.zoom, img_h / self.zoom
            
            x0 = self.view_offset[0]
            y0 = max(0, min(self.view_offset[1], img_h - view_h))
            self.view_offset[1] = y0

            x0_wrapped = x0 % img_w
            
            box1_x_end = min(img_w, x0_wrapped + view_w)
            box1 = (x0_wrapped, y0, box1_x_end, y0 + view_h)
            crop1 = image_with_overlays.crop(box1)
            
            stitched_image = Image.new('RGB', (int(view_w), int(view_h)))
            stitched_image.paste(crop1, (0, 0))
            
            if x0_wrapped + view_w > img_w:
                remaining_w = (x0_wrapped + view_w) - img_w
                box2 = (0, y0, remaining_w, y0 + view_h)
                crop2 = image_with_overlays.crop(box2)
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
            y_r1, z_r1 = ndc_y * cx - z * sx, ndc_y * sx + z * cx
            x_r2, z_r2 = ndc_x * cy - z_r1 * sy, ndc_x * sy + z_r1 * cy
            lat, lon = math.asin(-y_r1), math.atan2(x_r2, z_r2)
            map_x = int(((lon / math.pi) * 0.5 + 0.5) * self.generator.x_range)
            map_y = int(((-lat / math.pi) + 0.5) * self.generator.y_range)
            return map_x, map_y
            
        if not hasattr(self, 'generator') or not self.generator or not self.generator.pil_image: return -1, -1
        
        img_w, img_h = self.generator.pil_image.size
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w == 0 or canvas_h == 0: return -1, -1
        
        percent_x, percent_y = canvas_x / canvas_w, canvas_y / canvas_h
        view_w, view_h = img_w / self.zoom, img_h / self.zoom
        
        img_x = (self.view_offset[0] + (percent_x * view_w)) % img_w
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
        self.pan_start_pos = (event.x, event.y)
        if not self.adding_placemark.get():
            self.canvas.config(cursor="fleur")

    def _on_pan_end(self, event):
        self.pan_start_pos = None
        if not self.adding_placemark.get():
            self.canvas.config(cursor="")

    def _on_pan_move(self, event):
        if self.pan_start_pos is None or self.adding_placemark.get(): return
        
        dx, dy = event.x - self.pan_start_pos[0], event.y - self.pan_start_pos[1]
        
        if self.params['projection'].get() == 'Orthographic':
            self.params['rotation_y'].set(self.params['rotation_y'].get() + dx * 0.5)
            self.params['rotation_x'].set(max(-90, min(90, self.params['rotation_x'].get() + dy * 0.5)))
        else:
            self.view_offset[0] -= dx / self.zoom
            self.view_offset[1] -= dy / self.zoom
        
        self.pan_start_pos = (event.x, event.y)
        self.redraw_canvas()
        
    def toggle_placemark_mode(self):
        if self.adding_placemark.get():
            self.canvas.config(cursor="crosshair")
        else:
            self.canvas.config(cursor="")

    def save_image(self):
        if not self.generator or not self.generator.pil_image: return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")])
        if file_path:
            try:
                img_with_overlays = self._draw_hex_grid_on_image(self._draw_placemarks_on_image(self.generator.pil_image.copy()))
                img_with_overlays.save(file_path)

            except Exception as e: tk.messagebox.showerror("Save Error", f"Failed to save image:\n{e}")

    def save_preset(self):
        params_to_save = {key: var.get() for key, var in self.params.items()}
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
                self.redraw_canvas()

                self.on_projection_change()
                self.on_style_change()
            except Exception as e: tk.messagebox.showerror("Load Error", f"Failed to load preset:\n{e}")

    def open_palette_editor(self):
        PaletteEditor(self, self.apply_palette_from_editor).grab_set()

    def apply_palette_from_editor(self, new_palette):
        self.palette = new_palette
        self.recolor_map()
        
    def regenerate_ice(self):
        if not self.generator: return
        self.set_ui_state(is_generating=True)
        params_dict = {key: var.get() for key, var in self.params.items()}
        thread = threading.Thread(target=self.run_generation_in_thread, args=(params_dict,), daemon=True)
        thread.start()

    def recolor_map(self):
        if not self.generator or self.generator.color_map is None: return
        self.set_ui_state(is_generating=True)
        params_dict = {key: var.get() for key, var in self.params.items()}
        thread = threading.Thread(target=self.run_generation_in_thread, args=(params_dict,), daemon=True)
        thread.start()
        
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
        self.frame_viewer_frame.grid_remove()
        self.simulation_frames = []
        self.current_frame_index = -1
        
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
        
        original_ice_seed = self.params['ice_seed'].get()
        thaw_seed = self.params['thaw_ice_seed'].get()
        
        use_separate_thaw_seed = thaw_seed != original_ice_seed
        
        if event == 'Ice Age Cycle':
            peak_ice = 90.0
            peak_water = start_water - 10
            midpoint = frames / 2.0
            
            switched_to_thaw_noise = False
            
            for i in range(frames):
                self.update_generation_progress(int((i + 1) / frames * 100), f"Rendering frame {i+1}/{frames}...")
                
                if i < midpoint:
                    progress = i / midpoint
                    current_ice = start_ice + (peak_ice - start_ice) * progress
                    current_water = start_water + (peak_water - start_water) * progress
                else:
                    if use_separate_thaw_seed and not switched_to_thaw_noise:
                        self.generator.set_ice_seed(thaw_seed)
                        switched_to_thaw_noise = True
                    
                    progress = (i - midpoint) / midpoint
                    current_ice = peak_ice + (start_ice - peak_ice) * progress
                    current_water = peak_water + (start_water - peak_water) * progress

                self.generator.finalize_map(current_water, current_ice, self.params['map_style'].get())
                frame_image = self.generator.create_image()

                frame_data = {
                    'image': frame_image.copy(),
                    'color_map': self.generator.color_map.copy(),
                    'color_map_before_ice': self.generator.color_map_before_ice.copy()
                }
                self.simulation_frames.append(frame_data)
                
                self.generator.pil_image = frame_image
                self.after(0, self.redraw_canvas)

                frame_image_with_overlays = self._draw_hex_grid_on_image(self._draw_placemarks_on_image(frame_image))
                frame_image_with_overlays.save(os.path.join(save_dir, f"{base_filename}_{i+1:03d}.png"))
                time.sleep(0.05) 

        if use_separate_thaw_seed:
            self.generator.set_ice_seed(original_ice_seed)
        self.after(0, self.finalize_simulation)

    def finalize_simulation(self):
        if self.simulation_frames:
            self.current_frame_index = len(self.simulation_frames) - 1
            self.display_simulation_frame(self.current_frame_index)
            self.frame_viewer_frame.grid(row=self.frame_viewer_row, columnspan=3, sticky='ew', padx=5, pady=5)
        
        self.set_ui_state(is_generating=False)
        self.status_label.config(text="Ready")

    def show_previous_frame(self):
        if self.simulation_frames and self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.display_simulation_frame(self.current_frame_index)

    def show_next_frame(self):
        if self.simulation_frames and self.current_frame_index < len(self.simulation_frames) - 1:
            self.current_frame_index += 1
            self.display_simulation_frame(self.current_frame_index)

    def display_simulation_frame(self, index):
        if not self.simulation_frames or not (0 <= index < len(self.simulation_frames)):
            return

        frame_data = self.simulation_frames[index]
        self.generator.pil_image = frame_data['image']
        self.generator.color_map = frame_data['color_map']
        self.generator.color_map_before_ice = frame_data['color_map_before_ice']
        self.redraw_canvas()

        self.frame_label.config(text=f"{index + 1} / {len(self.simulation_frames)}")
        self.prev_frame_button.config(state=tk.NORMAL if index > 0 else tk.DISABLED)
        self.next_frame_button.config(state=tk.NORMAL if index < len(self.simulation_frames) - 1 else tk.DISABLED)

if __name__ == "__main__":
    app = App()
    app.mainloop()
