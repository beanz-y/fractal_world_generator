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
import numba # type: ignore
from queue import Queue

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
        self.layers = {
            'settlements': [],
            'placemarks': [],
            'natural_features': {'peaks': [], 'ranges': [], 'areas': [], 'bays': []}
        }
        self.layer_visibility = {
            'Settlements': tk.BooleanVar(value=True),
            'Placemarks': tk.BooleanVar(value=True),
            'Features': tk.BooleanVar(value=True)
        }
        
        self.placing_feature_info = None # Will store {'name': str, 'type': str}
        
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

        self.perspective_viewer_instance = None
        
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
            'num_features': tk.IntVar(value=10) # New parameter for POIs
        }
        self._create_control_widgets()
        self.tooltip = MapTooltip(self)
        self.canvas.bind("<Configure>", self.redraw_canvas)
        self.canvas.bind("<MouseWheel>", self._on_zoom)
        self.canvas.bind("<Button-1>", self._on_canvas_press)
        self.canvas.bind("<Button-3>", self._on_right_click) # Right-click to edit
        self.canvas.bind("<ButtonRelease-1>", self._on_pan_end)
        self.canvas.bind("<B1-Motion>", self._on_pan_move)
        self.canvas.bind("<Motion>", self._on_map_hover)
        self.canvas.bind("<Leave>", self._on_map_leave)
        
    def _create_control_widgets(self):
        row = 0
        self.controls_frame.grid_columnconfigure(1, weight=1)

        # Create a Notebook widget to hold the control tabs
        notebook = ttk.Notebook(self.controls_frame)
        notebook.grid(row=row, columnspan=3, sticky='ew', padx=5, pady=5); row += 1

        # Create frames for each tab
        gen_frame = ttk.Frame(notebook, padding="10")
        climate_frame = ttk.Frame(notebook, padding="10")
        civ_frame = ttk.Frame(notebook, padding="10")
        display_frame = ttk.Frame(notebook, padding="10")
        sim_frame = ttk.Frame(notebook, padding="10")

        notebook.add(gen_frame, text='Generation')
        notebook.add(climate_frame, text='Climate')
        notebook.add(civ_frame, text='Lore')
        notebook.add(display_frame, text='Display')
        notebook.add(sim_frame, text='Simulator')

        # --- Generation Parameters Tab ---
        gen_frame.grid_columnconfigure(1, weight=1)
        g_row = 0
        self._create_entry_widget("Width:", self.params['width'], g_row, master=gen_frame); g_row += 1
        self._create_entry_widget("Height:", self.params['height'], g_row, master=gen_frame); g_row += 1

        ttk.Label(gen_frame, text="World Scale:").grid(row=g_row, column=0, sticky='w', padx=5, pady=2);
        world_size_combo = ttk.Combobox(gen_frame, textvariable=self.params['world_size_preset'],
                                   values=['Kingdom (~1,600 km)', 'Region (~6,400 km)', 'Continent (~16,000 km)'],
                                   state="readonly")
        world_size_combo.grid(row=g_row, column=1, columnspan=2, sticky='ew', padx=5); g_row += 1
        world_size_combo.bind("<<ComboboxSelected>>", self.redraw_canvas)

        self._create_entry_widget("Seed:", self.params['seed'], g_row, include_random_button=True, master=gen_frame); g_row += 1
        self._create_slider_widget("Faults:", self.params['faults'], 1, 2000, g_row, master=gen_frame); g_row += 1
        self._create_slider_widget("Erosion:", self.params['erosion'], 0, 50, g_row, master=gen_frame); g_row += 1

        # --- Climate Parameters Tab ---
        climate_frame.grid_columnconfigure(1, weight=1)
        c_row = 0
        self._create_entry_widget("Ice Seed:", self.params['ice_seed'], c_row, include_random_button=True, master=climate_frame); c_row += 1
        self._create_slider_widget("Water %:", self.params['water'], 0, 100, c_row, master=climate_frame); c_row += 1
        self._create_slider_widget("Ice %:", self.params['ice'], 0, 100, c_row, master=climate_frame); c_row += 1
        self._create_slider_widget("Altitude Temp. Effect:", self.params['altitude_temp_effect'], 0, 1, c_row, master=climate_frame); c_row += 1

        ttk.Label(climate_frame, text="Wind Direction:").grid(row=c_row, column=0, columnspan=2, sticky='w', padx=5); c_row += 1
        wind_combo = ttk.Combobox(climate_frame, textvariable=self.params['wind_direction'],
                                  values=['West to East', 'East to West', 'North to South', 'South to North'], state="readonly")
        wind_combo.grid(row=c_row, column=0, columnspan=2, sticky='ew', padx=5, pady=(0,5)); c_row += 1

        # --- Civilization & Features Tab ---
        civ_frame.grid_columnconfigure(1, weight=1)
        civ_row = 0
        ttk.Label(civ_frame, text="Theme:").grid(row=civ_row, column=0, columnspan=2, sticky='w', padx=5); civ_row += 1
        theme_combo = ttk.Combobox(civ_frame, textvariable=self.params['theme'],
                                   values=['High Fantasy', 'Sci-Fi', 'Post-Apocalyptic', 'Feudal Japan', 'Lovecraftian'], state="readonly")
        theme_combo.grid(row=civ_row, column=0, columnspan=2, sticky='ew', padx=5, pady=(0,5)); civ_row += 1
        self._create_slider_widget("Settlements:", self.params['num_settlements'], 0, 500, civ_row, master=civ_frame); civ_row += 1
        self._create_slider_widget("Features:", self.params['num_features'], 0, 50, civ_row, master=civ_frame); civ_row += 1

        # --- Display Tab ---
        display_frame.grid_columnconfigure(1, weight=1)
        disp_row = 0
        style_frame = ttk.Frame(display_frame); style_frame.grid(row=disp_row, columnspan=2, sticky='ew', padx=5, pady=2); disp_row+=1
        ttk.Label(style_frame, text="Style:").pack(side=tk.LEFT)
        ttk.Radiobutton(style_frame, text="Biome", variable=self.params['map_style'], value="Biome", command=self.on_style_change).pack(side=tk.LEFT)
        ttk.Radiobutton(style_frame, text="Terrain", variable=self.params['map_style'], value="Terrain", command=self.on_style_change).pack(side=tk.LEFT)

        proj_frame = ttk.Frame(display_frame); proj_frame.grid(row=disp_row, columnspan=2, sticky='ew', padx=5, pady=2); disp_row+=1
        ttk.Label(proj_frame, text="Projection:").pack(side=tk.LEFT)
        ttk.Radiobutton(proj_frame, text="2D", variable=self.params['projection'], value="Equirectangular", command=self.on_projection_change).pack(side=tk.LEFT)
        ttk.Radiobutton(proj_frame, text="Globe", variable=self.params['projection'], value="Orthographic", command=self.on_projection_change).pack(side=tk.LEFT)

        self.rotation_frame = ttk.Labelframe(display_frame, text="Globe Rotation")
        self.rotation_frame.grid(row=disp_row, columnspan=2, sticky='ew', padx=5, pady=5); disp_row += 1
        self._create_slider_widget("Yaw:", self.params['rotation_y'], -180, 180, 0, master=self.rotation_frame)
        self._create_slider_widget("Pitch:", self.params['rotation_x'], -90, 90, 1, master=self.rotation_frame)
        self.params['rotation_y'].trace_add('write', lambda *_: self.redraw_canvas())
        self.params['rotation_x'].trace_add('write', lambda *_: self.redraw_canvas())

        ttk.Label(display_frame, text="Preset Palettes:").grid(row=disp_row, column=0, columnspan=2, sticky='w', padx=5); disp_row += 1
        self.palette_combobox = ttk.Combobox(display_frame, values=list(PREDEFINED_PALETTES.keys()), state="readonly")
        self.palette_combobox.set("Biome")
        self.palette_combobox.grid(row=disp_row, columnspan=2, sticky='ew', padx=5, pady=(0,10)); disp_row += 1
        self.palette_combobox.bind("<<ComboboxSelected>>", self.apply_predefined_palette)

        self.layers_frame = ttk.Labelframe(display_frame, text="Layers & Tools") # Renamed for clarity
        self.layers_frame.grid(row=disp_row, columnspan=2, sticky='ew', padx=5, pady=5); disp_row += 1
        self._create_layer_widgets()

        # --- Simulator Tab ---
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
        
        # --- Controls Below the Tabs ---
        self.progress = ttk.Progressbar(self.controls_frame, orient='horizontal', mode='determinate')
        self.progress.grid(row=row, columnspan=3, sticky='ew', pady=(5,0)); row += 1
        self.status_label = ttk.Label(self.controls_frame, text="Ready")
        self.status_label.grid(row=row, columnspan=3, sticky='ew', padx=5, pady=(0,5)); row += 1

        action_frame = ttk.Frame(self.controls_frame)
        action_frame.grid(row=row, columnspan=3, pady=5); row += 1
        self.generate_button = ttk.Button(action_frame, text="Regenerate World", command=self.start_generation)
        self.generate_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.random_button = ttk.Button(action_frame, text="New Random World", command=self.randomize_and_generate)
        self.random_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

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
        """Creates the layer visibility and tools control widgets."""
        for widget in self.layers_frame.winfo_children():
            widget.destroy()

        for name, var in self.layer_visibility.items():
            var.trace_add('write', lambda *_, n=name: self.redraw_canvas())

        ttk.Checkbutton(self.layers_frame, text="Settlements", variable=self.layer_visibility['Settlements']).grid(row=0, column=0, sticky='w', padx=5)
        ttk.Checkbutton(self.layers_frame, text="Features", variable=self.layer_visibility['Features']).grid(row=1, column=0, sticky='w', padx=5)
        
        placemark_frame = ttk.Frame(self.layers_frame)
        placemark_frame.grid(row=2, column=0, columnspan=2, sticky='ew')
        ttk.Checkbutton(placemark_frame, text="Placemarks", variable=self.layer_visibility['Placemarks']).pack(side=tk.LEFT, padx=5)
        ttk.Button(placemark_frame, text="Place Custom Feature...", command=self.open_placemark_dialog).pack(side=tk.LEFT, padx=5)
        
        # --- NEW --- Add perspective view button here
        perspective_frame = ttk.Frame(self.layers_frame)
        perspective_frame.grid(row=3, column=0, columnspan=2, sticky='w', pady=(5,0))
        ttk.Button(perspective_frame, text="Set Perspective View...", command=self.set_perspective_view).pack(side=tk.LEFT, padx=5)

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
        if self.placing_feature_info:
            self._add_placemark_at_click(event)
        else:
            self._on_pan_start(event)
    
    def _add_placemark_at_click(self, event):
        if not self.generator or not self.placing_feature_info: return

        # --- NEW --- Handle the perspective view placement
        if self.placing_feature_info.get('type') == 'PERSPECTIVE_VIEW':
            img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
            if img_x is not None:
                # Close the old viewer if it exists
                if self.perspective_viewer_instance and self.perspective_viewer_instance.winfo_exists():
                    self.perspective_viewer_instance.on_close()

                # Create the new viewer and store the instance
                self.perspective_viewer_instance = PerspectiveViewer(self, (img_x, img_y))
            
            self.placing_feature_info = None
            self.canvas.config(cursor="")
            self.status_label.config(text="Ready")
            return

        # Original placemark logic
        name = self.placing_feature_info['name']
        ptype = self.placing_feature_info['type']
        
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        if img_x is None: return

        new_feature = {'name': name, 'user_placed': True}

        # Handle Area Features
        if ptype in ["Ocean/Sea", "Mountain Range", "Forest Area", "Desert Area", "Jungle Area", "Tundra Area"]:
            new_feature['center'] = (img_x, img_y)
            if ptype == "Mountain Range":
                new_feature['type'] = 'range'
                self.layers['natural_features']['ranges'].append(new_feature)
            else:
                new_feature['type'] = 'area' 
                self.layers['natural_features']['areas'].append(new_feature)
        
        # Handle Point Features
        else:
            new_feature['x'], new_feature['y'] = img_x, img_y
            if ptype == 'Settlement':
                new_feature.update({'type': 'settlement', 'stype': 'custom'})
                self.layers['settlements'].append(new_feature)
            elif ptype == 'Mountain Peak':
                new_feature.update({'type': 'peak'})
                self.layers['natural_features']['peaks'].append(new_feature)
            elif ptype == 'Bay':
                 new_feature.update({'type': 'bay'})
                 self.layers['natural_features']['bays'].append(new_feature)
            else: # Generic Placemark
                new_feature.update({'type': 'placemark'})
                self.layers['placemarks'].append(new_feature)
        
        self.placing_feature_info = None
        self.canvas.config(cursor="")
        self.status_label.config(text="Ready")
        self.redraw_canvas()

    def _on_right_click(self, event):
        """Finds the nearest editable feature and opens a dialog to rename it."""
        if self.generator is None: return

        canvas_x, canvas_y = event.x, event.y
        
        all_items = []
        if self.layers.get('settlements'): all_items.extend(self.layers['settlements'])
        if self.layers.get('placemarks'): all_items.extend(self.layers['placemarks'])
        if self.layers.get('natural_features'):
            for key in ['peaks', 'bays']:
                if self.layers['natural_features'].get(key):
                    all_items.extend(self.layers['natural_features'][key])
            for key in ['areas', 'ranges']:
                 if self.layers['natural_features'].get(key):
                    for area in self.layers['natural_features'][key]:
                        area_proxy = area.copy()
                        area_proxy['x'], area_proxy['y'] = area['center']
                        all_items.append(area_proxy)
        
        if not all_items: return

        min_dist_sq = float('inf')
        closest_item_proxy = None
        
        for item in all_items:
            item_canvas_x, item_canvas_y = self.image_to_canvas_coords(item['x'], item['y'])
            if item_canvas_x is not None:
                dist_sq = (canvas_x - item_canvas_x)**2 + (canvas_y - item_canvas_y)**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_item_proxy = item

        if closest_item_proxy and min_dist_sq < 20**2:
            original_item = self.find_original_item(closest_item_proxy)
            if original_item:
                new_name = simpledialog.askstring("Edit Name", "Enter new name:", initialvalue=original_item['name'], parent=self)
                if new_name:
                    original_item['name'] = new_name
                    self.redraw_canvas()

    def find_original_item(self, item_proxy):
        """Finds the original dictionary for a given item proxy."""
        proxy_type = item_proxy.get('type')
        if not proxy_type: return None

        if proxy_type == 'settlement':
            for item in self.layers.get('settlements', []):
                if item['x'] == item_proxy['x'] and item['y'] == item_proxy['y']: return item
        elif proxy_type == 'placemark':
            for item in self.layers.get('placemarks', []):
                if item['x'] == item_proxy['x'] and item['y'] == item_proxy['y']: return item
        else: # Natural features
            for key in ['peaks', 'areas', 'bays', 'ranges']:
                for item in self.layers.get('natural_features', {}).get(key, []):
                    if 'center' in item and item['center'][0] == item_proxy['x'] and item['center'][1] == item_proxy['y']:
                        return item
                    elif 'center' not in item and item['x'] == item_proxy['x'] and item['y'] == item_proxy['y']:
                        return item
        return None

    def _on_map_hover(self, event):
        if self.placing_feature_info: return
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
        for widget in [self.generate_button, self.random_button, self.recolor_button, self.palette_button, self.save_button, self.palette_combobox, self.run_sim_button]:
            if widget not in [self.generate_button, self.random_button]:
                widget.config(state=tk.NORMAL if not is_generating and self.generator else tk.DISABLED)
        self.generate_button.config(state=state)
        self.random_button.config(state=state)

    def randomize_and_generate(self):
        self.params['seed'].set(random.randint(0, 100000))
        self.params['ice_seed'].set(random.randint(0, 100000))
        self.params['moisture_seed'].set(random.randint(0, 100000))
        self.start_generation()

    def start_generation(self):
        self.set_ui_state(is_generating=True)
        self.frame_viewer_frame.grid_remove()
        self.simulation_frames = []
        self.current_frame_index = -1
        self.layers = {
            'settlements': [],
            'placemarks': [],
            'natural_features': {'peaks': [], 'ranges': [], 'areas': [], 'bays': []}
        }
        self.zoom = 1.0
        self.view_offset = [0, 0]
        self.cached_globe_source_image = None # Invalidate cache
        params_dict = {key: var.get() for key, var in self.params.items() if key != 'world_size_preset'}
        
        size_preset = self.params['world_size_preset'].get()
        circumference_map = {
            'Kingdom (~1,600 km)': 1600.0,
            'Region (~6,400 km)': 6400.0,
            'Continent (~16,000 km)': 16000.0,
        }
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
        
        self.after(0, self.finalize_generation)

    def finalize_generation(self):
        self.layers['settlements'].extend(self.generator.settlements)
        for key, value in self.generator.natural_features.items():
            if key in self.layers['natural_features']:
                self.layers['natural_features'][key].extend(value)
        
        self.cached_globe_source_image = None
        
        self.redraw_canvas()
        self.set_ui_state(is_generating=False)
        self.status_label.config(text="Ready")

    def update_generation_progress(self, value, text):
        self.after(0, lambda: (
            self.progress.config(value=value),
            self.status_label.config(text=text)
        ))

    def _create_text_with_outline(self, x, y, text, **kwargs):
        """Helper to draw text with an outline on the Tkinter canvas."""
        outline_color = 'black'
        # Create a copy of the kwargs for the outline, and set the fill color
        outline_kwargs = kwargs.copy()
        outline_kwargs['fill'] = outline_color
        
        # Draw outline text (the "stroke")
        self.canvas.create_text(x-1, y, text=text, **outline_kwargs)
        self.canvas.create_text(x+1, y, text=text, **outline_kwargs)
        self.canvas.create_text(x, y-1, text=text, **outline_kwargs)
        self.canvas.create_text(x, y+1, text=text, **outline_kwargs)

        # Draw main text on top using the original kwargs
        self.canvas.create_text(x, y, text=text, **kwargs)

    def _draw_layers_on_image(self, image):
        """Draws vector layers directly onto the supplied PIL Image object."""
        draw = ImageDraw.Draw(image, 'RGBA')
        
        try:
            font_l = ImageFont.truetype("arialbd.ttf", size=16)
            font_m = ImageFont.truetype("arial.ttf", size=12)
            font_s = ImageFont.truetype("arial.ttf", size=10)
        except IOError:
            font_l = ImageFont.load_default()
            font_m = ImageFont.load_default()
            font_s = ImageFont.load_default()

        # Draw Features
        if self.layer_visibility['Features'].get() and 'natural_features' in self.layers:
            # Draw areas and ranges first
            for area in self.layers['natural_features'].get('areas', []):
                cx, cy = self.image_to_canvas_coords(*area['center'])
                if cx is not None: draw.text((cx, cy), area['name'], font=font_l, fill=(255, 255, 220, 192), anchor="mm", stroke_width=2, stroke_fill=(0,0,0,128))
            for r in self.layers['natural_features'].get('ranges', []):
                cx, cy = self.image_to_canvas_coords(*r['center'])
                if cx is not None: draw.text((cx, cy), r['name'], font=font_l, fill=(220, 200, 180, 192), anchor="mm", stroke_width=2, stroke_fill=(0,0,0,128))

            # Draw smaller point-based features on top
            for bay in self.layers['natural_features'].get('bays', []):
                bx, by = self.image_to_canvas_coords(bay['x'], bay['y'])
                if bx is not None: draw.text((bx, by), bay['name'], font=font_m, fill=(200, 220, 255, 192), anchor="mm", stroke_width=2, stroke_fill=(0,0,0,128))
            for peak in self.layers['natural_features'].get('peaks', []):
                px, py = self.image_to_canvas_coords(peak['x'], peak['y'])
                if px is not None:
                    draw.polygon([(px, py-6), (px-4, py+3), (px+4, py+3)], fill=(200, 200, 200, 128), outline='black')
                    draw.text((px, py + 5), peak['name'], font=font_s, fill="white", anchor="ms", stroke_width=1, stroke_fill="black")

        # Draw Settlements
        if self.layer_visibility['Settlements'].get() and 'settlements' in self.layers:
            icon_radius = 3
            for settlement in self.layers['settlements']:
                canvas_x, canvas_y = self.image_to_canvas_coords(settlement['x'], settlement['y'])
                if canvas_x is not None:
                    draw.ellipse((canvas_x - icon_radius, canvas_y - icon_radius, canvas_x + icon_radius, canvas_y + icon_radius), fill='white', outline='black')
                    draw.text((canvas_x + icon_radius + 2, canvas_y), settlement['name'], font=font_s, fill="white", anchor="lm", stroke_width=1, stroke_fill="black")

        # Draw Placemarks
        if self.layer_visibility['Placemarks'].get() and 'placemarks' in self.layers:
            icon_radius = 4
            for pm in self.layers['placemarks']:
                canvas_x, canvas_y = self.image_to_canvas_coords(pm['x'], pm['y'])
                if canvas_x is not None:
                    draw.ellipse((canvas_x - icon_radius, canvas_y - icon_radius, canvas_x + icon_radius, canvas_y + icon_radius), fill='red', outline='black')
                    draw.text((canvas_x + icon_radius + 2, canvas_y), pm['name'], font=font_m, fill="white", anchor="lm", stroke_width=1, stroke_fill="black")

    def _draw_scale_bar(self):
        """Draws a dynamic scale bar on the canvas for the 2D view."""
        self.canvas.delete("scale_bar")
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()

        if canvas_w <= 1 or self.generator is None: return

        circumference_km = self.world_circumference_km
        zoom = self.zoom
        img_h = self.generator.y_range
        
        center_y_map_px = self.view_offset[1] + (img_h / zoom) / 2
        center_lat_rad = (center_y_map_px / img_h - 0.5) * math.pi
        
        if math.cos(center_lat_rad) <= 0: return
        km_per_canvas_pixel = (circumference_km * math.cos(center_lat_rad)) / (zoom * canvas_w)

        if km_per_canvas_pixel <= 0: return

        target_bar_width_px = canvas_w / 5
        target_distance_km = target_bar_width_px * km_per_canvas_pixel
        
        if target_distance_km == 0: return
        magnitude = 10 ** math.floor(math.log10(target_distance_km))
        residual = target_distance_km / magnitude
        
        if residual < 1.5: nice_distance_km = 1 * magnitude
        elif residual < 3.5: nice_distance_km = 2 * magnitude
        elif residual < 7.5: nice_distance_km = 5 * magnitude
        else: nice_distance_km = 10 * magnitude

        final_bar_width_px = nice_distance_km / km_per_canvas_pixel

        margin = 20
        bar_x = margin
        bar_y = canvas_h - margin
        
        self._create_text_with_outline(bar_x + final_bar_width_px / 2, bar_y - 10, text=f"{int(nice_distance_km)} km", fill="white", tags="scale_bar", anchor="s")
        self.canvas.create_line(bar_x, bar_y - 5, bar_x, bar_y + 5, fill="white", tags="scale_bar", width=2)
        self.canvas.create_line(bar_x, bar_y, bar_x + final_bar_width_px, bar_y, fill="white", tags="scale_bar", width=2)
        self.canvas.create_line(bar_x + final_bar_width_px, bar_y - 5, bar_x + final_bar_width_px, bar_y + 5, fill="white", tags="scale_bar", width=2)

    def redraw_canvas(self, event=None):
        if not hasattr(self, 'generator') or not self.generator or not self.generator.pil_image: return
        self.tooltip.hide()
        
        self.canvas.delete("all")
        
        projection = self.params['projection'].get()

        if projection == 'Orthographic':
            source_array = np.array(self.generator.pil_image.convert('RGB'))
            canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if canvas_w <= 1 or canvas_h <= 1: return

            rot_y, rot_x = self.params['rotation_y'].get(), self.params['rotation_x'].get()
            new_img_array = render_globe(source_array, canvas_w, canvas_h, rot_y, rot_x, self.generator.x_range, self.generator.y_range)
            final_image = Image.fromarray(new_img_array).convert('RGBA')
            
            self._draw_layers_on_image(final_image)
            
            self.tk_image = ImageTk.PhotoImage(final_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image, tags="map_image")

        else: # Equirectangular view
            img_w, img_h = self.generator.pil_image.size
            view_w, view_h = img_w / self.zoom, img_h / self.zoom
            
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
            
            self._draw_layers_on_canvas_2d()
            self._draw_scale_bar()

    def _draw_layers_on_canvas_2d(self):
        """Draws vector layers directly onto the canvas for the 2D view."""
        self.canvas.delete("overlay")

        # Draw Features
        if self.layer_visibility['Features'].get() and 'natural_features' in self.layers:
            # Draw large area names first
            for area in self.layers['natural_features'].get('areas', []):
                cx, cy = self.image_to_canvas_coords(*area['center'])
                if cx is not None: self._create_text_with_outline(cx, cy, text=area['name'], font=("Arial", 14, "italic"), fill="#DDEEFF", anchor="c", tags="overlay")
            
            for r in self.layers['natural_features'].get('ranges', []):
                cx, cy = self.image_to_canvas_coords(*r['center'])
                if cx is not None: self._create_text_with_outline(cx, cy, text=r['name'], font=("Arial", 12, "bold"), fill="#D2B48C", anchor="c", tags="overlay")

            # Draw smaller, more specific feature names
            for bay in self.layers['natural_features'].get('bays', []):
                bx, by = self.image_to_canvas_coords(bay['x'], bay['y'])
                if bx is not None: self._create_text_with_outline(bx, by, text=bay['name'], font=("Arial", 10, "italic"), fill="#add8e6", anchor="c", tags="overlay")
            
            for peak in self.layers['natural_features'].get('peaks', []):
                px, py = self.image_to_canvas_coords(peak['x'], peak['y'])
                if px is not None:
                    self._create_text_with_outline(px, py-8, text=peak['name'], font=("Arial", 9), fill="white", anchor="s", tags="overlay")
                    self.canvas.create_text(px, py, text="▲", font=("Arial", 12, "bold"), fill="white", tags="overlay")
                    self.canvas.create_text(px, py, text="▲", font=("Arial", 12), fill="black", tags="overlay")


        # Draw Settlements
        if self.layer_visibility['Settlements'].get() and 'settlements' in self.layers:
            icon_radius = 3
            for settlement in self.layers['settlements']:
                canvas_x, canvas_y = self.image_to_canvas_coords(settlement['x'], settlement['y'])
                if canvas_x is not None:
                    self.canvas.create_oval(canvas_x - icon_radius, canvas_y - icon_radius, canvas_x + icon_radius, canvas_y + icon_radius, fill='white', outline='black', tags="overlay")
                    self._create_text_with_outline(canvas_x + icon_radius + 2, canvas_y, text=settlement['name'], anchor='w', font=("Arial", 9), fill="white", tags="overlay")

        # Draw Placemarks
        if self.layer_visibility['Placemarks'].get() and 'placemarks' in self.layers:
            icon_radius = 4
            for pm in self.layers['placemarks']:
                canvas_x, canvas_y = self.image_to_canvas_coords(pm['x'], pm['y'])
                if canvas_x is not None:
                    self.canvas.create_oval(canvas_x - icon_radius, canvas_y - icon_radius, canvas_x + icon_radius, canvas_y + icon_radius, fill='red', outline='black', tags="overlay")
                    self._create_text_with_outline(canvas_x + icon_radius + 2, canvas_y, text=pm['name'], anchor='w', font=("Arial", 10), fill="white", tags="overlay")

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
        if self.placing_feature_info: return
        self.pan_start_pos = (event.x, event.y)
        self.canvas.config(cursor="fleur")

    def _on_pan_end(self, event):
        self.pan_start_pos = None
        if not self.placing_feature_info:
            self.canvas.config(cursor="")

    def _on_pan_move(self, event):
        if self.pan_start_pos is None or self.placing_feature_info: return
        
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
        
    def open_placemark_dialog(self):
        dialog = PlacemarkDialog(self)
        if dialog.result:
            self.placing_feature_info = dialog.result
            self.canvas.config(cursor="crosshair")
            self.status_label.config(text=f"Click on the map to place a {self.placing_feature_info['type']}...")

    def save_image(self):
        if not self.generator or not self.generator.pil_image: return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")], title="Save Image As")
        if not file_path: return
        
        # To save what's on screen, we need to regenerate the currently displayed image
        current_image = self.canvas.postscript(colormode='color')
        img = Image.open(io.BytesIO(current_image.encode('utf-8')))
        img.save(file_path)

    def save_preset(self):
        params_to_save = {key: var.get() for key, var in self.params.items()}
        # Intentionally do not save layers, as they should be regenerated
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
                
                # Clear existing layers and image, as they are now stale
                self.layers = {
                    'settlements': [],
                    'placemarks': [],
                    'natural_features': {'peaks': [], 'ranges': [], 'areas': [], 'bays': []}
                }
                self.pil_image = None
                self.generator = None
                self.canvas.delete("all")
                self.status_label.config(text="Preset loaded. Click 'Regenerate World'.")

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
                frame_with_overlays = frame_image 
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

    def open_profile_viewer(self):
        if not self.generator or self.generator.world_map is None:
            tk.messagebox.showwarning("No Map Data", "Please generate a world before creating a profile view.")
            return

        y_coord = simpledialog.askinteger(
            "Select Profile Row",
            f"Enter the Y-coordinate (latitude) to create a profile from (0 to {self.generator.y_range - 1}):",
            minvalue=0,
            maxvalue=self.generator.y_range - 1
        )

        if y_coord is not None:
            ProfileViewer(self, y_coord)

    def draw_perspective_fov(self, camera_pos, angle_deg, fov_deg=72, length=50):
        """Draws the Field of View cone on the main canvas."""
        self.canvas.delete("fov_lines")

        if self.params['projection'].get() != 'Equirectangular':
            return

        cam_img_x, cam_img_y = camera_pos
        cam_canvas_x, cam_canvas_y = self.image_to_canvas_coords(cam_img_x, cam_img_y)

        if cam_canvas_x is None:
            return

        angle_rad = math.radians(angle_deg)
        fov_rad = math.radians(fov_deg)

        left_angle = angle_rad - fov_rad / 2
        right_angle = angle_rad + fov_rad / 2

        # --- CORRECTION: Negate the sine component to flip the X-axis ---
        end_left_x = cam_img_x - math.sin(left_angle) * length
        end_left_y = cam_img_y + math.cos(left_angle) * length
        end_right_x = cam_img_x - math.sin(right_angle) * length
        end_right_y = cam_img_y + math.cos(right_angle) * length

        end_left_canvas_x, end_left_canvas_y = self.image_to_canvas_coords(end_left_x, end_left_y)
        end_right_canvas_x, end_right_canvas_y = self.image_to_canvas_coords(end_right_x, end_right_y)

        if end_left_canvas_x is not None:
            self.canvas.create_line(
                cam_canvas_x, cam_canvas_y, end_left_canvas_x, end_left_canvas_y,
                fill="yellow", width=2, tags="fov_lines"
            )
        if end_right_canvas_x is not None:
            self.canvas.create_line(
                cam_canvas_x, cam_canvas_y, end_right_canvas_x, end_right_canvas_y,
                fill="yellow", width=2, tags="fov_lines"
            )
        self.canvas.create_oval(
            cam_canvas_x - 4, cam_canvas_y - 4, cam_canvas_x + 4, cam_canvas_y + 4,
            fill="yellow", outline="black", tags="fov_lines"
        )

    def clear_perspective_fov(self):
        """Removes the FOV lines from the canvas."""
        self.canvas.delete("fov_lines")

    def set_perspective_view(self):
        """Prepares the app to place a camera for the 3D perspective view."""
        if not self.generator or not self.generator.pil_image:
            tk.messagebox.showwarning("No Map", "Please generate a map first.")
            return

        # If a viewer is already open, bring it to the front instead of creating a new one
        if self.perspective_viewer_instance and self.perspective_viewer_instance.winfo_exists():
            self.perspective_viewer_instance.lift()
            return
            
        self.placing_feature_info = {'type': 'PERSPECTIVE_VIEW'}
        self.canvas.config(cursor="crosshair")
        self.status_label.config(text="Click on the map to set the camera position for the 3D view...")

@numba.jit(nopython=True)
def _numba_scalar_clip(value, min_val, max_val):
    """A numba-compatible function to clip a single scalar value."""
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    else:
        return value
    
@numba.jit(nopython=True, fastmath=True)
def _numba_render_perspective(height_map, color_map, palette, camera_pos, view_angle_rad, fov, width, height, max_distance, camera_altitude_offset, vertical_exaggeration):
    """
    Performs the core ray casting calculation with configurable vertical exaggeration.
    """
    # vertical_exaggeration is now passed in as a parameter
    map_h, map_w = height_map.shape
    cam_x, cam_y = camera_pos

    safe_cam_y = _numba_scalar_clip(int(cam_y), 0, map_h - 1)
    safe_cam_x = _numba_scalar_clip(int(cam_x), 0, map_w - 1)

    # Use the passed-in vertical_exaggeration
    cam_height = (height_map[safe_cam_y, safe_cam_x] * vertical_exaggeration) + camera_altitude_offset

    image_buffer = np.zeros((height, width, 3), dtype=np.uint8)

    sky_color = np.array([135, 206, 235], dtype=np.uint8)
    fog_color = np.array([170, 185, 195], dtype=np.uint8)
    horizon = height // 2

    for y in range(height):
        if y < horizon:
            image_buffer[y, :, :] = sky_color
        else:
            interp = min(1.0, (y - horizon) / horizon)
            r = int(sky_color[0]*(1-interp) + fog_color[0]*interp)
            g = int(sky_color[1]*(1-interp) + fog_color[1]*interp)
            b = int(sky_color[2]*(1-interp) + fog_color[2]*interp)
            image_buffer[y, :, 0], image_buffer[y, :, 1], image_buffer[y, :, 2] = r,g,b

    water_level = np.percentile(height_map.ravel(), 60.0)

    for i in range(width):
        y_buffer = height

        angle = view_angle_rad - fov / 2 + fov * i / width
        sin_angle = -math.sin(angle)
        cos_angle = math.cos(angle)

        for z in range(1, int(max_distance)):
            map_x = int(cam_x + sin_angle * z)
            map_y = int(cam_y + cos_angle * z)

            map_x_wrapped = map_x % map_w
            if not (0 <= map_y < map_h): continue

            terrain_height = height_map[map_y, map_x_wrapped]
            color_index = color_map[map_y, map_x_wrapped]

            is_water = 1 <= color_index <= 7 and terrain_height < water_level

            # Use the passed-in vertical_exaggeration
            display_height = water_level if is_water else terrain_height
            display_height *= vertical_exaggeration

            base_color = palette[color_index]

            projected_height = int((cam_height - display_height) / z * 240 + horizon)

            if projected_height >= y_buffer: continue

            fog_factor = min(1.0, (z / max_distance)**1.5)
            r = int(base_color[0]*(1-fog_factor) + fog_color[0]*fog_factor)
            g = int(base_color[1]*(1-fog_factor) + fog_color[1]*fog_factor)
            b = int(base_color[2]*(1-fog_factor) + fog_color[2]*fog_factor)

            draw_start_y = _numba_scalar_clip(projected_height, 0, height)
            draw_end_y = _numba_scalar_clip(y_buffer, 0, height)

            for y in range(draw_start_y, draw_end_y):
                image_buffer[y, i, 0] = r
                image_buffer[y, i, 1] = g
                image_buffer[y, i, 2] = b

            y_buffer = projected_height
            if y_buffer <= 0: break

    return image_buffer

class PerspectiveViewer(tk.Toplevel):
    def __init__(self, parent, camera_pos):
        super().__init__(parent)
        self.parent = parent
        self.generator = parent.generator
        self.camera_pos = camera_pos
        self.view_angle = 180.0
        self.image = None
        self.tk_image = None
        self.rendering_thread = None
        self._is_rendering = threading.Event()

        # --- NEW: Variables for UI controls ---
        self.camera_altitude = tk.DoubleVar(value=40.0)
        self.vertical_exaggeration = tk.DoubleVar(value=2.5)

        self.title(f"Perspective View from {self.camera_pos}")
        self.geometry("800x680") # Made window taller for the new slider

        self.render_queue = Queue()

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(main_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.display_image(rerender=False))

        controls_frame = ttk.Frame(self)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)

        self.pan_left_button = ttk.Button(controls_frame, text="Pan Left", command=lambda: self.pan(-15.0))
        self.pan_left_button.pack(side=tk.LEFT, padx=5)
        self.pan_right_button = ttk.Button(controls_frame, text="Pan Right", command=lambda: self.pan(15.0))
        self.pan_right_button.pack(side=tk.LEFT, padx=5)
        self.save_button = ttk.Button(controls_frame, text="Save Image...", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(side=tk.RIGHT, padx=5)

        # --- NEW: Control Sliders Frame ---
        sliders_frame = ttk.Frame(self)
        sliders_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        sliders_frame.columnconfigure(1, weight=1)

        ttk.Label(sliders_frame, text="Altitude:").grid(row=0, column=0, sticky='w', padx=5)
        altitude_slider = ttk.Scale(
            sliders_frame, from_=5, to=500, orient='horizontal',
            variable=self.camera_altitude, command=self.on_slider_change
        )
        altitude_slider.grid(row=0, column=1, sticky='ew', padx=5)

        ttk.Label(sliders_frame, text="Exaggeration:").grid(row=1, column=0, sticky='w', padx=5)
        exaggeration_slider = ttk.Scale(
            sliders_frame, from_=1.0, to=15.0, orient='horizontal',
            variable=self.vertical_exaggeration, command=self.on_slider_change
        )
        exaggeration_slider.grid(row=1, column=1, sticky='ew', padx=5)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.update_parent_fov()
        self.start_render_thread()
        self.check_render_queue()

    def on_slider_change(self, value):
        """Callback for when any slider is moved."""
        if not self._is_rendering.is_set():
            self.start_render_thread()

    def update_parent_fov(self):
        self.parent.draw_perspective_fov(self.camera_pos, self.view_angle)

    def on_close(self):
        self._is_rendering.set()
        self.parent.clear_perspective_fov()
        self.parent.perspective_viewer_instance = None
        self.destroy()

    def pan(self, angle_change):
        if self._is_rendering.is_set(): return
        self.view_angle += angle_change
        self.update_parent_fov()
        self.start_render_thread()

    def start_render_thread(self):
        self._is_rendering.set()
        self.save_button.config(state=tk.DISABLED)
        self.pan_left_button.config(state=tk.DISABLED)
        self.pan_right_button.config(state=tk.DISABLED)

        self.canvas.delete("render_text")
        self.canvas.create_text(
            self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2,
            text="Rendering...", fill="yellow", font=("Arial", 24, "bold"), tags="render_text"
        )

        self.rendering_thread = threading.Thread(target=self.render_view_threaded, daemon=True)
        self.rendering_thread.start()

    def render_view_threaded(self):
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 10 or height < 10: width, height = 800, 600

        supersampling_factor = 2
        render_w = width * supersampling_factor
        render_h = height * supersampling_factor

        fov = math.pi / 2.5
        max_distance = self.generator.x_range / 1.5
        palette_np = np.array(self.generator.palette, dtype=np.uint8)

        # Get the current values from the Tkinter variables
        current_altitude = self.camera_altitude.get()
        current_exaggeration = self.vertical_exaggeration.get()

        image_data = _numba_render_perspective(
            self.generator.world_map.T, self.generator.color_map.T, palette_np,
            self.camera_pos, math.radians(self.view_angle),
            fov, render_w, render_h, max_distance,
            camera_altitude_offset=current_altitude,
            vertical_exaggeration=current_exaggeration # Pass the new exaggeration here
        )

        if self._is_rendering.is_set():
            high_res_image = Image.fromarray(image_data, 'RGB')
            final_image = high_res_image.resize((width, height), Image.Resampling.LANCZOS)
            self.render_queue.put(final_image)

    def check_render_queue(self):
        try:
            new_image = self.render_queue.get_nowait()
            self.image = new_image
            self.display_image(rerender=True)
            self._is_rendering.clear()
            self.save_button.config(state=tk.NORMAL)
            self.pan_left_button.config(state=tk.NORMAL)
            self.pan_right_button.config(state=tk.NORMAL)
        except Exception:
            pass
        finally:
            if self.winfo_exists():
                self.after(100, self.check_render_queue)

    def display_image(self, rerender=True):
        if self.image:
            if rerender:
                self.canvas.delete("all")
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def save_image(self):
        if self.image:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png", filetypes=[("PNG Image", "*.png")],
                title="Save Perspective View As"
            )
            if file_path: self.image.save(file_path)

class PlacemarkDialog(simpledialog.Dialog):
    def body(self, master):
        self.title("Place Custom Feature")
        
        self.name_var = tk.StringVar()
        self.type_var = tk.StringVar(value="Placemark")

        ttk.Label(master, text="Feature Name:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.name_entry = ttk.Entry(master, textvariable=self.name_var)
        self.name_entry.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        
        ttk.Label(master, text="Feature Type:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        
        feature_types = [
            "Placemark", "Settlement", "Mountain Peak", "Bay", 
            "Ocean/Sea", "Mountain Range", "Forest Area", "Desert Area", "Jungle Area", "Tundra Area"
        ]
        self.type_combo = ttk.Combobox(master, textvariable=self.type_var, 
                                       values=feature_types,
                                       state="readonly")
        self.type_combo.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        
        self.result = None
        return self.name_entry # initial focus

    def apply(self):
        name = self.name_var.get()
        ptype = self.type_var.get()
        if name and ptype:
            self.result = {'name': name, 'type': ptype}

if __name__ == "__main__":
    app = App()
    app.mainloop()