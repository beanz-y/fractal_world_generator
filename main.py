# beanz-y/fractal_world_generator/fractal_world_generator-8b752999818ebdee7e3c696935b618f2a364ff8f/main.py
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

# Add the script's directory to the Python path to ensure local modules can be found.
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from world_generator import FractalWorldGenerator
from utils import MapTooltip, PaletteEditor
from constants import PREDEFINED_PALETTES, BIOME_DEFINITIONS
from projection import render_globe, canvas_coords_to_map, map_coords_to_canvas

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
        self.world_circumference_km = 1600.0 # Default value
        
        # --- Layer Management System ---
        self.layers = {}
        self.layer_visibility = {}
        self.cached_globe_source_image = None
        
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
            'world_size_preset': tk.StringVar(value='Kingdom (~1,600 km)'),
            'seed': tk.IntVar(value=random.randint(0, 100000)),
            'ice_seed': tk.IntVar(value=random.randint(0, 100000)),
            'moisture_seed': tk.IntVar(value=random.randint(0, 100000)),
            'thaw_ice_seed': tk.IntVar(value=random.randint(0, 100000)),
            'map_style': tk.StringVar(value='Biome'),
            'projection': tk.StringVar(value='Equirectangular'),
            'rotation_x': tk.DoubleVar(value=0.0),
            'rotation_y': tk.DoubleVar(value=0.0),
            'faults': tk.IntVar(value=200),
            'erosion': tk.IntVar(value=5),
            'water': tk.DoubleVar(value=60.0),
            'ice': tk.DoubleVar(value=15.0),
            'altitude_temp_effect': tk.DoubleVar(value=0.5),
            'wind_direction': tk.StringVar(value='West to East'), 
            'simulation_event': tk.StringVar(value='Ice Age Cycle'),
            'simulation_speed': tk.StringVar(value='Realistic (Ease In/Out)'),
            'simulation_frames': tk.IntVar(value=20),
            'theme': tk.StringVar(value='High Fantasy'), 
            'num_settlements': tk.IntVar(value=50),
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
        
        # World Size Preset Dropdown
        ttk.Label(gen_frame, text="World Scale:").grid(row=g_row, column=0, sticky='w', padx=5, pady=2);
        world_size_combo = ttk.Combobox(gen_frame, textvariable=self.params['world_size_preset'], 
                                   values=['Kingdom (~1,600 km)', 'Region (~6,400 km)', 'Continent (~16,000 km)'], 
                                   state="readonly")
        world_size_combo.grid(row=g_row, column=1, columnspan=2, sticky='ew', padx=5); g_row += 1
        world_size_combo.bind("<<ComboboxSelected>>", self.redraw_canvas)


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

        # --- Civilization Parameters ---
        civ_frame = ttk.Labelframe(self.controls_frame, text="Civilization")
        civ_frame.grid(row=row, columnspan=3, sticky='ew', padx=5, pady=5); row += 1
        civ_frame.grid_columnconfigure(1, weight=1)
        
        civ_row = 0
        ttk.Label(civ_frame, text="Theme:").grid(row=civ_row, column=0, columnspan=2, sticky='w', padx=5); civ_row += 1
        theme_combo = ttk.Combobox(civ_frame, textvariable=self.params['theme'], 
                                   values=['High Fantasy', 'Sci-Fi', 'Post-Apocalyptic'], state="readonly")
        theme_combo.grid(row=civ_row, column=0, columnspan=2, sticky='ew', padx=5, pady=(0,5)); civ_row += 1
        self._create_slider_widget("Settlements:", self.params['num_settlements'], 0, 500, civ_row, master=civ_frame); civ_row += 1

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

        # --- Layers Frame ---
        self.layers_frame = ttk.Labelframe(self.controls_frame, text="Layers")
        self.layers_frame.grid(row=row, columnspan=3, sticky='ew', padx=5, pady=5); row += 1
        self._create_layer_widgets()
        
        sim_frame = ttk.Labelframe(self.controls_frame, text="Age Simulator")
        sim_frame.grid(row=row, columnspan=3, sticky='ew', padx=5, pady=5); row += 1
        sim_frame.grid_columnconfigure(1, weight=1)

        sf_row = 0
        ttk.Label(sim_frame, text="Event:").grid(row=sf_row, column=0, sticky='w', padx=5)
        event_combo = ttk.Combobox(sim_frame, textvariable=self.params['simulation_event'], values=['Ice Age Cycle'], state="readonly")
        event_combo.grid(row=sf_row, column=1, columnspan=2, sticky='ew', padx=5); sf_row += 1
        
        ttk.Label(sim_frame, text="Speed:").grid(row=sf_row, column=0, sticky='w', padx=5)
        speed_combo = ttk.Combobox(sim_frame, textvariable=self.params['simulation_speed'], values=['Linear', 'Realistic (Ease In/Out)'], state="readonly")
        speed_combo.grid(row=sf_row, column=1, columnspan=2, sticky='ew', padx=5); sf_row += 1

        self._create_entry_widget("Thaw Ice Seed:", self.params['thaw_ice_seed'], sf_row, include_random_button=True, master=sim_frame); sf_row += 1
        self._create_entry_widget("Frames:", self.params['simulation_frames'], sf_row, master=sim_frame); sf_row += 1
        
        self.run_sim_button = ttk.Button(sim_frame, text="Run Simulation", command=self.start_age_simulation, state=tk.DISABLED)
        self.run_sim_button.grid(row=sf_row, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
                
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

    def _create_layer_widgets(self):
        # Clear existing widgets if any
        for widget in self.layers_frame.winfo_children():
            widget.destroy()

        self.layer_visibility = {
            'Settlements': tk.BooleanVar(value=True),
            'Placemarks': tk.BooleanVar(value=True),
        }
        
        # Bind toggling redraw to each variable
        for name, var in self.layer_visibility.items():
            var.trace_add('write', lambda *args, n=name: self._toggle_layer_visibility(n))

        # Create checkbuttons
        ttk.Checkbutton(self.layers_frame, text="Settlements", variable=self.layer_visibility['Settlements']).grid(row=0, column=0, sticky='w', padx=5)
        
        placemark_frame = ttk.Frame(self.layers_frame)
        placemark_frame.grid(row=1, column=0, columnspan=2, sticky='w')
        ttk.Checkbutton(placemark_frame, text="Placemarks", variable=self.layer_visibility['Placemarks']).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(placemark_frame, text="Add Placemark", variable=self.adding_placemark, command=self.toggle_placemark_mode).pack(side=tk.LEFT, padx=5)

    def _toggle_layer_visibility(self, layer_name):
        self.redraw_canvas()

    def ease_in_out_cubic(self, x):
        """Non-linear animation timing function."""
        return 4 * x * x * x if x < 0.5 else 1 - pow(-2 * x + 2, 3) / 2

    def _create_entry_widget(self, label_text, var, row, include_random_button=False, master=None, use_grid=False):
        if master is None: master = self.controls_frame
        
        container = ttk.Frame(master)
        
        ttk.Label(container, text=label_text).pack(side=tk.LEFT, padx=(0,5))
        entry = ttk.Entry(container, textvariable=var, width=10)
        entry.pack(side=tk.LEFT)
        if include_random_button:
            button = ttk.Button(container, text="🎲", width=3, command=lambda v=var: v.set(random.randint(0, 100000)))
            button.pack(side=tk.LEFT, padx=(5,0))
            
        if use_grid:
            container.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        else:
            container.grid(row=row, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
        
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
            if 'placemarks' not in self.layers:
                self.layers['placemarks'] = []
            self.layers['placemarks'].append({'name': placemark_name, 'x': img_x, 'y': img_y})
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
            if 56 <= index <= 59:
                return "Alpine Glacier"
            if 52 <= index <= 55:
                return "Polar Ice"
            
            for name, props in BIOME_DEFINITIONS.items():
                if props['idx'] <= index < props['idx'] + props.get('shades', 1):
                    if name in ['glacier', 'alpine_glacier']: 
                        continue
                    return name.replace('_', ' ').title()
        
        if 1 <= index <= 7: return "Water"
        if 16 <= index <= 31: return "Land"
        return "Unknown"
        
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
        self.layers = {} # Reset layers
        self.zoom = 1.0
        self.view_offset = [0, 0]
        self.cached_globe_source_image = None # Invalidate cache
        params_dict = {key: var.get() for key, var in self.params.items() if key != 'world_size_preset'}
        
        # Convert world size preset string to circumference value
        size_preset = self.params['world_size_preset'].get()
        circumference_map = {
            'Kingdom (~1,600 km)': 1600.0,
            'Region (~6,400 km)': 6400.0,
            'Continent (~16,000 km)': 16000.0,
        }
        # --- FIX: Store circumference in the app instance ---
        self.world_circumference_km = circumference_map.get(size_preset, 6400.0)
        params_dict['world_circumference_km'] = self.world_circumference_km

        thread = threading.Thread(target=self.run_generation_in_thread, args=(params_dict,), daemon=True)
        thread.start()
        
    def run_generation_in_thread(self, params_dict):
        map_style = params_dict.get('map_style', 'Biome')
        palette_name = 'Biome' if map_style == 'Biome' else 'Default'
        self.palette = list(PREDEFINED_PALETTES[palette_name])

        self.generator = FractalWorldGenerator(params_dict, self.palette, self.update_generation_progress)
        self.pil_image = self.generator.generate()
        
        # After base generation, generate civilization layers
        self.generator.generate_settlements()
        
        self.after(0, self.finalize_generation)

    def finalize_generation(self):
        # Populate layer data from the generator
        self.layers['settlements'] = self.generator.settlements
        self.cached_globe_source_image = None # Invalidate cache
        
        self.redraw_canvas()
        self.set_ui_state(is_generating=False)
        self.status_label.config(text="Ready")

    def update_generation_progress(self, value, text):
        self.after(0, lambda: (
            self.progress.config(value=value),
            self.status_label.config(text=text)
        ))

    def _draw_layers_on_image(self, image):
        """Draws vector layers directly onto the supplied PIL Image object."""
        draw = ImageDraw.Draw(image)
        
        # Load a font
        try:
            settlement_font = ImageFont.truetype("arial.ttf", size=9)
            placemark_font = ImageFont.truetype("arial.ttf", size=10)
        except IOError:
            settlement_font = ImageFont.load_default()
            placemark_font = ImageFont.load_default()

        # Draw Settlements
        if self.layer_visibility.get('Settlements', tk.BooleanVar(value=True)).get() and 'settlements' in self.layers:
            icon_radius = 3
            for settlement in self.layers['settlements']:
                # Use image_to_canvas_coords to find where the settlement should be on the final image
                canvas_x, canvas_y = self.image_to_canvas_coords(settlement['x'], settlement['y'])
                if canvas_x is not None and canvas_y is not None:
                    box = (canvas_x - icon_radius, canvas_y - icon_radius, canvas_x + icon_radius, canvas_y + icon_radius)
                    draw.ellipse(box, fill='white', outline='black')
                    # Draw text label
                    text_pos = (canvas_x + icon_radius + 2, canvas_y)
                    draw.text(text_pos, settlement['name'], font=settlement_font, fill="white", anchor="lm")

        # Draw Placemarks
        if self.layer_visibility.get('Placemarks', tk.BooleanVar(value=False)).get() and 'placemarks' in self.layers:
            icon_radius = 4
            for pm in self.layers['placemarks']:
                canvas_x, canvas_y = self.image_to_canvas_coords(pm['x'], pm['y'])
                if canvas_x is not None and canvas_y is not None:
                    box = (canvas_x - icon_radius, canvas_y - icon_radius, canvas_x + icon_radius, canvas_y + icon_radius)
                    draw.ellipse(box, fill='red', outline='black')
                    draw.text((canvas_x + icon_radius + 2, canvas_y), pm['name'], font=placemark_font, fill="white", anchor="lm")
    
    def _draw_scale_bar(self):
        """Draws a dynamic scale bar on the canvas for the 2D view."""
        self.canvas.delete("scale_bar")
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()

        if canvas_w <= 1 or self.generator is None:
            return

        # --- FIX: Read circumference from the app instance attribute ---
        circumference_km = self.world_circumference_km
        zoom = self.zoom
        img_h = self.generator.y_range
        
        # Calculate latitude at the center of the view to account for projection distortion
        center_y_map_px = self.view_offset[1] + (img_h / zoom) / 2
        center_lat_rad = (center_y_map_px / img_h - 0.5) * math.pi
        
        # Calculate km per canvas pixel at the view's central latitude
        km_per_canvas_pixel = (circumference_km * math.cos(center_lat_rad)) / (zoom * canvas_w)

        if km_per_canvas_pixel <= 0:
            return

        # Find a "nice" distance and pixel width for the bar
        target_bar_width_px = canvas_w / 5
        target_distance_km = target_bar_width_px * km_per_canvas_pixel
        
        # Algorithm to find a "nice" round number for the scale distance
        if target_distance_km == 0: return
        magnitude = 10 ** math.floor(math.log10(target_distance_km))
        residual = target_distance_km / magnitude
        
        if residual < 1.5: nice_distance_km = 1 * magnitude
        elif residual < 3.5: nice_distance_km = 2 * magnitude
        elif residual < 7.5: nice_distance_km = 5 * magnitude
        else: nice_distance_km = 10 * magnitude

        final_bar_width_px = nice_distance_km / km_per_canvas_pixel

        # Draw the bar in the bottom-left corner
        margin = 20
        bar_x = margin
        bar_y = canvas_h - margin
        
        self.canvas.create_line(bar_x, bar_y - 5, bar_x, bar_y + 5, fill="white", tags="scale_bar", width=2)
        self.canvas.create_line(bar_x, bar_y, bar_x + final_bar_width_px, bar_y, fill="white", tags="scale_bar", width=2)
        self.canvas.create_line(bar_x + final_bar_width_px, bar_y - 5, bar_x + final_bar_width_px, bar_y + 5, fill="white", tags="scale_bar", width=2)
        self.canvas.create_text(bar_x + final_bar_width_px / 2, bar_y - 10, text=f"{int(nice_distance_km)} km", fill="white", tags="scale_bar", anchor="s")

    def redraw_canvas(self, event=None):
        if not hasattr(self, 'generator') or not self.generator or not self.generator.pil_image: return
        self.tooltip.hide()
        
        self.canvas.delete("all")
        
        projection = self.params['projection'].get()

        if projection == 'Orthographic':
            # Render the base globe
            source_array = np.array(self.generator.pil_image.convert('RGB'))
            canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if canvas_w <= 1 or canvas_h <= 1: return

            rot_y = self.params['rotation_y'].get()
            rot_x = self.params['rotation_x'].get()

            new_img_array = render_globe(source_array, canvas_w, canvas_h, rot_y, rot_x, self.generator.x_range, self.generator.y_range)
            final_image = Image.fromarray(new_img_array)
            
            # Draw layers directly onto the PIL image
            self._draw_layers_on_image(final_image)
            
            # Display the final composite image
            self.tk_image = ImageTk.PhotoImage(final_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image, tags="map_image")

        else: # Equirectangular view
            img_w, img_h = self.generator.pil_image.size
            view_w = img_w / self.zoom
            view_h = img_h / self.zoom
            
            x0 = self.view_offset[0]
            y0 = max(0, min(self.view_offset[1], img_h - view_h))
            self.view_offset[1] = y0
            x0_wrapped = x0 % img_w
            
            box1 = (x0_wrapped, y0, min(img_w, x0_wrapped + view_w), y0 + view_h)
            crop1 = self.generator.pil_image.crop(box1)
            
            stitched_image = Image.new('RGB', (int(view_w), int(view_h)))
            stitched_image.paste(crop1, (0, 0))
            
            if x0_wrapped + view_w > img_w:
                remaining_w = (x0_wrapped + view_w) - img_w
                box2 = (0, y0, remaining_w, y0 + view_h)
                crop2 = self.generator.pil_image.crop(box2)
                stitched_image.paste(crop2, (crop1.width, 0))
            
            canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if canvas_w <= 1 or canvas_h <=1: return
            
            resized_image = stitched_image.resize((canvas_w, canvas_h), Image.Resampling.NEAREST)
            self.tk_image = ImageTk.PhotoImage(resized_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image, tags="map_image")
            
            # For 2D view, draw overlays and scale bar on the canvas
            self._draw_layers_on_canvas_2d()
            self._draw_scale_bar()

    def _draw_layers_on_canvas_2d(self):
        """Draws vector layers directly onto the canvas for the 2D view."""
        self.canvas.delete("overlay")

        if self.layer_visibility.get('Settlements', tk.BooleanVar(value=True)).get() and 'settlements' in self.layers:
            font_size = 9
            icon_radius = 3
            for settlement in self.layers['settlements']:
                canvas_x, canvas_y = self.image_to_canvas_coords(settlement['x'], settlement['y'])
                if canvas_x is not None and canvas_y is not None:
                    self.canvas.create_oval(canvas_x - icon_radius, canvas_y - icon_radius, 
                                            canvas_x + icon_radius, canvas_y + icon_radius, 
                                            fill='white', outline='black', tags="overlay")
                    self.canvas.create_text(canvas_x + icon_radius + 2, canvas_y,
                                            text=settlement['name'], anchor='w', font=("Arial", font_size), fill="white", tags="overlay")

        if self.layer_visibility.get('Placemarks', tk.BooleanVar(value=False)).get() and 'placemarks' in self.layers:
            font_size = 10
            icon_radius = 4
            for pm in self.layers['placemarks']:
                canvas_x, canvas_y = self.image_to_canvas_coords(pm['x'], pm['y'])
                if canvas_x is not None and canvas_y is not None:
                    self.canvas.create_oval(canvas_x - icon_radius, canvas_y - icon_radius, 
                                        canvas_x + icon_radius, canvas_y + icon_radius, 
                                        fill='red', outline='black', tags="overlay")
                    self.canvas.create_text(canvas_x + icon_radius + 2, canvas_y,
                                        text=pm['name'], anchor='w', font=("Arial", font_size), fill="white", tags="overlay")

    def image_to_canvas_coords(self, img_x, img_y):
        """Converts full map image coordinates to on-screen canvas coordinates for the current projection."""
        if not hasattr(self, 'generator') or not self.generator.pil_image: return None, None
        
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1: return None, None

        if self.params['projection'].get() == 'Orthographic':
            return map_coords_to_canvas(
                img_x, img_y, canvas_w, canvas_h,
                self.params['rotation_y'].get(), self.params['rotation_x'].get(),
                self.generator.x_range, self.generator.y_range
            )
        else: # Equirectangular
            img_w, img_h = self.generator.pil_image.size
            view_w, view_h = img_w / self.zoom, img_h / self.zoom
            
            dx = img_x - self.view_offset[0]
            if abs(dx) > img_w / 2: dx -= np.sign(dx) * img_w
            dy = img_y - self.view_offset[1]
            
            canvas_x, canvas_y = (dx / view_w) * canvas_w, (dy / view_h) * canvas_h
            
            if 0 <= canvas_x <= canvas_w and 0 <= canvas_y <= canvas_h:
                return canvas_x, canvas_y
            return None, None
        
    def canvas_to_image_coords(self, canvas_x, canvas_y):
        if not hasattr(self, 'generator') or not self.generator or not self.generator.pil_image: return None, None

        if self.params['projection'].get() == 'Orthographic':
            canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
            return canvas_coords_to_map(
                canvas_x, canvas_y, canvas_w, canvas_h,
                self.params['rotation_y'].get(), self.params['rotation_x'].get(),
                self.generator.x_range, self.generator.y_range
            )
        else:
            img_w, img_h = self.generator.pil_image.size
            canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if canvas_w == 0 or canvas_h == 0: return None, None
            
            percent_x, percent_y = canvas_x / canvas_w, canvas_y / canvas_h
            view_w = img_w / self.zoom
            view_h = img_h / self.zoom
            
            img_x = self.view_offset[0] + (percent_x * view_w)
            img_y = self.view_offset[1] + (percent_y * view_h)
            
            img_x_wrapped = img_x % img_w
            
            return int(img_x_wrapped), int(img_y)

    def _on_zoom(self, event):
        if not self.generator or not self.generator.pil_image: return
        if self.params['projection'].get() == 'Orthographic': return
        
        img_x_before, img_y_before = self.canvas_to_image_coords(event.x, event.y)
        if img_x_before is None: return

        factor = 1.1 if event.delta > 0 else 0.9
        self.zoom *= factor
        self.zoom = max(0.1, min(self.zoom, 50))

        img_w, img_h = self.generator.pil_image.size
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        
        view_w_after = img_w / self.zoom
        view_h_after = img_h / self.zoom

        self.view_offset[0] = img_x_before - (event.x / canvas_w) * view_w_after
        self.view_offset[1] = img_y_before - (event.y / canvas_h) * view_h_after

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
            self.params['rotation_y'].set(self.params['rotation_y'].get() - dx * 0.3)
            self.params['rotation_x'].set(max(-90, min(90, self.params['rotation_x'].get() - dy * 0.3)))
        else:
            img_w, img_h = self.generator.pil_image.size
            canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()

            px_per_canvas_x = (img_w / self.zoom) / canvas_w
            px_per_canvas_y = (img_h / self.zoom) / canvas_h
            
            delta_img_x = dx * px_per_canvas_x
            delta_img_y = dy * px_per_canvas_y
            
            self.view_offset[0] -= delta_img_x
            self.view_offset[1] -= delta_img_y
        
        self.pan_start_pos = (event.x, event.y)
        self.redraw_canvas()
        
    def toggle_placemark_mode(self):
        if self.adding_placemark.get():
            self.canvas.config(cursor="crosshair")
        else:
            self.canvas.config(cursor="")

    def save_image(self):
        # This function will need to be re-implemented for the new rendering system
        tk.messagebox.showinfo("Export Not Ready", "The export functionality is being updated.")
    def save_preset(self):
        params_to_save = {key: var.get() for key, var in self.params.items()}
        params_to_save['layers'] = self.layers
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
                
                self.layers = loaded_params.get('layers', {})
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
            max_temp_offset = -0.15 # Max cooling at peak ice age
            midpoint = frames / 2.0
            
            switched_to_thaw_noise = False
            
            for i in range(frames):
                self.update_generation_progress(int((i + 1) / frames * 100), f"Rendering frame {i+1}/{frames}...")
                
                raw_progress = (i / midpoint) if i < midpoint else ((i - midpoint) / midpoint)
                
                if self.params['simulation_speed'].get() == 'Realistic (Ease In/Out)':
                    eased_progress = self.ease_in_out_cubic(raw_progress)
                else:
                    eased_progress = raw_progress

                if i < midpoint:
                    current_ice = start_ice + (peak_ice - start_ice) * eased_progress
                    current_water = start_water + (peak_water - start_water) * eased_progress
                    current_temp_offset = max_temp_offset * eased_progress
                else:
                    if use_separate_thaw_seed and not switched_to_thaw_noise:
                        self.generator.set_ice_seed(thaw_seed)
                        switched_to_thaw_noise = True
                    
                    current_ice = peak_ice - (peak_ice - start_ice) * eased_progress
                    current_water = peak_water - (peak_water - start_water) * eased_progress
                    current_temp_offset = max_temp_offset * (1.0 - eased_progress)

                self.generator.finalize_map(current_water, current_ice, self.params['map_style'].get(), temp_offset=current_temp_offset)
                frame_image = self.generator.create_image()

                frame_data = {
                    'image': frame_image.copy(),
                    'color_map': self.generator.color_map.copy(),
                    'color_map_before_ice': self.generator.color_map_before_ice.copy()
                }
                self.simulation_frames.append(frame_data)
                
                self.generator.pil_image = frame_image
                # The _composite_layers method was not provided, so it is commented out.
                # frame_with_overlays = self._composite_layers(frame_image, is_export=True, export_label_size='Medium')
                frame_with_overlays = frame_image # Placeholder
                self.after(0, self.redraw_canvas)

                frame_with_overlays.save(os.path.join(save_dir, f"{base_filename}_{i+1:03d}.png"))
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

class ExportDialog(simpledialog.Dialog):
    def body(self, master):
        self.title("Export Options")
        
        self.export_type = tk.StringVar(value="VTT Map")
        self.label_size = tk.StringVar(value="Medium")

        ttk.Label(master, text="Export Type:").grid(row=0, sticky='w', padx=5, pady=2)
        ttk.Radiobutton(master, text="VTT Map (Overlays with fixed size)", variable=self.export_type, value="VTT Map").grid(row=1, sticky='w', padx=10)
        ttk.Radiobutton(master, text="GM Map (Overlays match current view)", variable=self.export_type, value="GM Map").grid(row=2, sticky='w', padx=10)
        ttk.Radiobutton(master, text="Player Map (No overlays)", variable=self.export_type, value="Player Map").grid(row=3, sticky='w', padx=10)

        self.label_frame = ttk.Labelframe(master, text="Label Size for VTT Export")
        self.label_frame.grid(row=4, columnspan=2, sticky='ew', padx=5, pady=5)
        
        ttk.Radiobutton(self.label_frame, text="Small", variable=self.label_size, value="Small").pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Radiobutton(self.label_frame, text="Medium", variable=self.label_size, value="Medium").pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Radiobutton(self.label_frame, text="Large", variable=self.label_size, value="Large").pack(side=tk.LEFT, padx=5, pady=5)
        
        self.export_type.trace_add('write', self._toggle_label_options)
        self.result = None
        return None

    def _toggle_label_options(self, *args):
        if self.export_type.get() == "VTT Map":
            for child in self.label_frame.winfo_children():
                child.config(state=tk.NORMAL)
        else:
            for child in self.label_frame.winfo_children():
                child.config(state=tk.DISABLED)

    def apply(self):
        self.result = {
            'type': self.export_type.get(),
            'label_size': self.label_size.get()
        }

if __name__ == "__main__":
    app = App()
    app.mainloop()