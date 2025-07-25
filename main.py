# beanz-y/fractal_world_generator/fractal_world_generator-28f75751b57dacf83432892d2293f1e3754a3ba6/main.py
#
# --- CHANGELOG ---
# 1. Revert: Plate Tectonics model UI has been removed.
# 2. Feature: UI for Climate and Rivers
#    - Added a "Season" dropdown to the Climate tab to apply global temperature modifiers.
#    - Added a "Number of Rivers" slider to the Lore tab.
#    - Added a "Rivers" checkbox to the Layers & Tools panel to toggle their visibility.
# 3. Refactor: Dynamic UI
#    - The main `start_generation` function now passes all relevant parameters to the generator.
# 4. Feature: UI for Lake Controls
#    - Added a "Lake Coverage %" slider to the Lore tab to control the percentage of potential lakes that form.
#    - Added a "Named Lakes" slider to the Lore tab to control the maximum number of lakes that receive names.
# 5. Bugfix: Corrected tooltip logic to display specific names for named water bodies (lakes, oceans, seas).
# -----------------

import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import random
import math
import threading
import json
import os
import sys
import time
from queue import Queue

# Add the script's directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import local modules
from world_generator import FractalWorldGenerator
from constants import PREDEFINED_PALETTES, BIOME_DEFINITIONS, THEME_NAME_FRAGMENTS
from projection import render_globe, canvas_coords_to_map, map_coords_to_canvas
from ui_components import MapTooltip, PaletteEditor
from dialogs import PlacemarkDialog
from viewers import PerspectiveViewer

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
        self.world_circumference_km = 1600.0
        
        self.layers = {
            'settlements': [], 'placemarks': [], 'rivers': [], 'borders': [],
            'natural_features': {'peaks': [], 'ranges': [], 'areas': []}
        }
        self.layer_visibility = {
            'Settlements': tk.BooleanVar(value=True),
            'Placemarks': tk.BooleanVar(value=True),
            'Features': tk.BooleanVar(value=True),
            'Rivers': tk.BooleanVar(value=True),
            'Borders': tk.BooleanVar(value=True)
        }
        
        self.placing_feature_info = None
        self.perspective_viewer_instance = None
        
        self.edit_mode = tk.BooleanVar(value=False)
        self.selected_item = None
        self.is_dragging = False
        self.drag_start_pos = None

        self.simulation_frames = []
        self.current_frame_index = -1
        self.feature_name_lookup = {}

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
            'season': tk.StringVar(value='Normal'),
            'simulation_event': tk.StringVar(value='Ice Age Cycle'),
            'simulation_speed': tk.StringVar(value='Realistic (Ease In/Out)'),
            'simulation_frames': tk.IntVar(value=20),
            'theme': tk.StringVar(value='High Fantasy'), 
            'num_settlements': tk.IntVar(value=50),
            'num_features': tk.IntVar(value=10),
            'num_rivers': tk.IntVar(value=50),
            'lake_coverage': tk.DoubleVar(value=50.0),
            'max_lakes_to_name': tk.IntVar(value=20),
            'num_kingdoms': tk.IntVar(value=8),
            'mountain_cost': tk.DoubleVar(value=8.0),
            'hill_cost': tk.DoubleVar(value=3.0)
        }
        self._create_control_widgets()
        self.tooltip = MapTooltip(self)
        self.canvas.bind("<Configure>", self.redraw_canvas)
        self.canvas.bind("<MouseWheel>", self._on_zoom)
        self.canvas.bind("<Button-1>", self._on_canvas_press)
        self.canvas.bind("<ButtonRelease-1>", self._on_pan_end)
        self.canvas.bind("<B1-Motion>", self._on_pan_move)
        self.canvas.bind("<Motion>", self._on_map_hover)
        self.canvas.bind("<Leave>", self._on_map_leave)

    def _draw_borders_on_canvas_2d(self):
        if not self.layer_visibility['Borders'].get() or 'borders' not in self.layers or not self.layers['borders']:
            return
        
        total_points = sum(len(k.get('border_points', [])) for k in self.layers['borders'])
        print(f"DEBUG: Drawing {len(self.layers['borders'])} borders with a total of {total_points} points.")

        for kingdom in self.layers['borders']:
            color = kingdom.get('color', 'white')
            for p in kingdom.get('border_points', []):
                # p[0] is x, p[1] is y
                cx, cy = self.image_to_canvas_coords(p[0], p[1])
                if cx is not None:
                    # --- THIS IS THE FIX ---
                    # Use the correct canvas coordinates (cx, cy) to draw the oval.
                    self.canvas.create_oval(cx - 1, cy - 1, cx + 1, cy + 1, fill=color, outline="", tags="overlay")

    def _create_control_widgets(self):
        row = 0
        self.controls_frame.grid_columnconfigure(1, weight=1)

        notebook = ttk.Notebook(self.controls_frame)
        notebook.grid(row=row, columnspan=3, sticky='ew', padx=5, pady=5); row += 1

        gen_frame = ttk.Frame(notebook, padding="10")
        climate_frame = ttk.Frame(notebook, padding="10")
        civ_frame = ttk.Frame(notebook, padding="10")
        display_frame = ttk.Frame(notebook, padding="10")
        sim_frame = ttk.Frame(notebook, padding="10")

        notebook.add(gen_frame, text='Generation'); notebook.add(climate_frame, text='Climate'); notebook.add(civ_frame, text='Lore'); notebook.add(display_frame, text='Display'); notebook.add(sim_frame, text='Simulator')

        # --- Generation Tab ---
        gen_frame.grid_columnconfigure(1, weight=1)
        g_row = 0
        self._create_entry_widget("Width:", self.params['width'], g_row, master=gen_frame); g_row += 1
        self._create_entry_widget("Height:", self.params['height'], g_row, master=gen_frame); g_row += 1

        ttk.Label(gen_frame, text="World Scale:").grid(row=g_row, column=0, sticky='w', padx=5, pady=2);
        world_size_combo = ttk.Combobox(gen_frame, textvariable=self.params['world_size_preset'], values=['Kingdom (~1,600 km)', 'Region (~6,400 km)', 'Continent (~16,000 km)'], state="readonly")
        world_size_combo.grid(row=g_row, column=1, columnspan=2, sticky='ew', padx=5); g_row += 1
        world_size_combo.bind("<<ComboboxSelected>>", self.redraw_canvas)

        self._create_entry_widget("Seed:", self.params['seed'], g_row, include_random_button=True, master=gen_frame); g_row += 1
        self._create_slider_widget("Faults:", self.params['faults'], 1, 2000, g_row, master=gen_frame); g_row += 1
        self._create_slider_widget("Erosion:", self.params['erosion'], 0, 50, g_row, master=gen_frame); g_row += 1

        # --- Climate Tab ---
        climate_frame.grid_columnconfigure(1, weight=1)
        c_row = 0
        self._create_entry_widget("Ice Seed:", self.params['ice_seed'], c_row, include_random_button=True, master=climate_frame); c_row += 1
        self._create_slider_widget("Water %:", self.params['water'], 0, 100, c_row, master=climate_frame); c_row += 1
        self._create_slider_widget("Ice %:", self.params['ice'], 0, 100, c_row, master=climate_frame); c_row += 1
        self._create_slider_widget("Altitude Temp. Effect:", self.params['altitude_temp_effect'], 0, 1, c_row, master=climate_frame); c_row += 1

        ttk.Label(climate_frame, text="Wind Direction:").grid(row=c_row, column=0, columnspan=2, sticky='w', padx=5); c_row += 1
        wind_combo = ttk.Combobox(climate_frame, textvariable=self.params['wind_direction'], values=['West to East', 'East to West', 'North to South', 'South to North'], state="readonly")
        wind_combo.grid(row=c_row, column=0, columnspan=2, sticky='ew', padx=5, pady=(0,5)); c_row += 1

        ttk.Label(climate_frame, text="Season:").grid(row=c_row, column=0, columnspan=2, sticky='w', padx=5); c_row += 1
        season_combo = ttk.Combobox(climate_frame, textvariable=self.params['season'], values=['Normal', 'Warm', 'Cold', 'Ice Age'], state="readonly")
        season_combo.grid(row=c_row, column=0, columnspan=2, sticky='ew', padx=5, pady=(0,5)); c_row += 1

        # --- Lore/Civilization Tab ---
        civ_frame.grid_columnconfigure(1, weight=1)
        civ_row = 0
        ttk.Label(civ_frame, text="Theme:").grid(row=civ_row, column=0, columnspan=2, sticky='w', padx=5); civ_row += 1
        theme_combo = ttk.Combobox(civ_frame, textvariable=self.params['theme'], values=list(THEME_NAME_FRAGMENTS.keys()), state="readonly")
        theme_combo.grid(row=civ_row, column=0, columnspan=2, sticky='ew', padx=5, pady=(0,5)); civ_row += 1
        self._create_slider_widget("Settlements:", self.params['num_settlements'], 0, 500, civ_row, master=civ_frame); civ_row += 1
        self._create_slider_widget("Features:", self.params['num_features'], 0, 50, civ_row, master=civ_frame); civ_row += 1
        self._create_slider_widget("Rivers:", self.params['num_rivers'], 0, 200, civ_row, master=civ_frame); civ_row += 1
        self._create_slider_widget("Lake Coverage %:", self.params['lake_coverage'], 0, 100, civ_row, master=civ_frame); civ_row += 1
        self._create_slider_widget("Named Lakes:", self.params['max_lakes_to_name'], 0, 50, civ_row, master=civ_frame); civ_row += 1

        politics_frame = ttk.Labelframe(civ_frame, text="Politics", padding=5)
        politics_frame.grid(row=civ_row, column=0, columnspan=2, sticky='ew', padx=5, pady=(10,0)); civ_row += 1
        politics_frame.columnconfigure(1, weight=1)
        p_row = 0
        self._create_slider_widget("Kingdoms:", self.params['num_kingdoms'], 2, 50, p_row, master=politics_frame); p_row += 1
        self._create_slider_widget("Mtn. Cost:", self.params['mountain_cost'], 1.0, 20.0, p_row, master=politics_frame); p_row += 1
        self._create_slider_widget("Hill Cost:", self.params['hill_cost'], 1.0, 10.0, p_row, master=politics_frame); p_row += 1

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
        self.params['rotation_y'].trace_add('write', lambda *_: self.redraw_canvas()); self.params['rotation_x'].trace_add('write', lambda *_: self.redraw_canvas())

        ttk.Label(display_frame, text="Preset Palettes:").grid(row=disp_row, column=0, columnspan=2, sticky='w', padx=5); disp_row += 1
        self.palette_combobox = ttk.Combobox(display_frame, values=list(PREDEFINED_PALETTES.keys()), state="readonly")
        self.palette_combobox.set("Biome")
        self.palette_combobox.grid(row=disp_row, columnspan=2, sticky='ew', padx=5, pady=(0,10)); disp_row += 1
        self.palette_combobox.bind("<<ComboboxSelected>>", self.apply_predefined_palette)

        self.layers_frame = ttk.Labelframe(display_frame, text="Layers & Tools")
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
        
        # --- Main Action Buttons ---
        self.progress = ttk.Progressbar(self.controls_frame, orient='horizontal', mode='determinate')
        self.progress.grid(row=row, columnspan=3, sticky='ew', pady=(5,0)); row += 1
        self.status_label = ttk.Label(self.controls_frame, text="Ready")
        self.status_label.grid(row=row, columnspan=3, sticky='ew', padx=5, pady=(0,5)); row += 1

        action_frame = ttk.Frame(self.controls_frame); action_frame.grid(row=row, columnspan=3, pady=5); row += 1
        self.generate_button = ttk.Button(action_frame, text="Regenerate World", command=self.start_generation); self.generate_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.random_button = ttk.Button(action_frame, text="New Random World", command=self.randomize_and_generate); self.random_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        recolor_frame = ttk.Frame(self.controls_frame); recolor_frame.grid(row=row, columnspan=3, pady=5); row += 1
        self.recolor_button = ttk.Button(recolor_frame, text="Recolor Map", command=self.recolor_map, state=tk.DISABLED); self.recolor_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.palette_button = ttk.Button(recolor_frame, text="Edit Palette", command=self.open_palette_editor); self.palette_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        file_frame = ttk.Labelframe(self.controls_frame, text="File Operations", padding=5); file_frame.grid(row=row, columnspan=3, pady=5, sticky="ew"); row += 1
        file_frame.columnconfigure(0, weight=1); file_frame.columnconfigure(1, weight=1)

        load_save_frame = ttk.Frame(file_frame); load_save_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Button(load_save_frame, text="Load Preset", command=self.load_preset).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(load_save_frame, text="Save Preset", command=self.save_preset).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        export_frame = ttk.Frame(file_frame); export_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5,0))
        self.save_button = ttk.Button(export_frame, text="Save Image As...", command=self.save_image, state=tk.DISABLED); self.save_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.save_layers_button = ttk.Button(export_frame, text="Save as Layers...", command=self.save_layered_image, state=tk.DISABLED); self.save_layers_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.frame_viewer_row = row; row += 1
        self.frame_viewer_frame = ttk.Labelframe(self.controls_frame, text="Frame Viewer")
        frame_nav_frame = ttk.Frame(self.frame_viewer_frame); frame_nav_frame.pack(pady=5)
        self.prev_frame_button = ttk.Button(frame_nav_frame, text="< Prev", command=self.show_previous_frame, state=tk.DISABLED); self.prev_frame_button.pack(side=tk.LEFT, padx=5)
        self.frame_label = ttk.Label(frame_nav_frame, text="0 / 0"); self.frame_label.pack(side=tk.LEFT, padx=5)
        self.next_frame_button = ttk.Button(frame_nav_frame, text="Next >", command=self.show_next_frame, state=tk.DISABLED); self.next_frame_button.pack(side=tk.LEFT, padx=5)
        
        self.on_projection_change()

    def _create_layer_widgets(self):
        for widget in self.layers_frame.winfo_children(): widget.destroy()
        for name, var in self.layer_visibility.items(): var.trace_add('write', lambda *_, n=name: self.redraw_canvas())
        
        ttk.Checkbutton(self.layers_frame, text="Settlements", variable=self.layer_visibility['Settlements']).grid(row=0, column=0, sticky='w', padx=5)
        ttk.Checkbutton(self.layers_frame, text="Features", variable=self.layer_visibility['Features']).grid(row=1, column=0, sticky='w', padx=5)
        ttk.Checkbutton(self.layers_frame, text="Rivers", variable=self.layer_visibility['Rivers']).grid(row=2, column=0, sticky='w', padx=5)

        ttk.Checkbutton(self.layers_frame, text="Borders", variable=self.layer_visibility['Borders']).grid(row=3, column=0, sticky='w', padx=5)

        placemark_frame = ttk.Frame(self.layers_frame); placemark_frame.grid(row=4, column=0, columnspan=2, sticky='w') # Changed row to 4
        ttk.Checkbutton(placemark_frame, text="Placemarks", variable=self.layer_visibility['Placemarks']).pack(side=tk.LEFT, padx=5)
        ttk.Button(placemark_frame, text="Place Custom Feature...", command=self.open_placemark_dialog).pack(side=tk.LEFT, padx=5)
        
        self.regen_borders_button = ttk.Button(self.layers_frame, text="Regenerate Borders", command=self.regenerate_borders, state=tk.DISABLED)
        self.regen_borders_button.grid(row=0, column=1, padx=5, sticky='e') # Place it on the right

        perspective_frame = ttk.Frame(self.layers_frame); perspective_frame.grid(row=5, column=0, columnspan=2, sticky='w', pady=(5,0)) # Changed row to 5
        ttk.Button(perspective_frame, text="Set Perspective View...", command=self.set_perspective_view).pack(side=tk.LEFT, padx=5)

        edit_frame = ttk.Labelframe(self.layers_frame, text="Editor"); edit_frame.grid(row=6, column=0, columnspan=2, sticky='ew', pady=(10,0), padx=5) # Changed row to 6
        ttk.Checkbutton(edit_frame, text="Edit Mode", variable=self.edit_mode, command=self._toggle_edit_mode).pack(side=tk.LEFT, padx=5)
        self.selection_info_frame = ttk.Frame(edit_frame)
        self.selection_name_label = ttk.Label(self.selection_info_frame, text="", wraplength=150)
        self.selection_rename_button = ttk.Button(self.selection_info_frame, text="Rename", command=self._rename_selected_item)
        self.selection_delete_button = ttk.Button(self.selection_info_frame, text="Delete", command=self._delete_selected_item)

    def _toggle_edit_mode(self):
        if not self.edit_mode.get(): self._deselect_item(); self.canvas.config(cursor="")
        else: self.canvas.config(cursor="hand2")

    def _update_selection_info_panel(self):
        if self.selected_item:
            self.selection_info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
            self.selection_name_label.config(text=self.selected_item['name']); self.selection_name_label.pack(side=tk.LEFT)
            self.selection_rename_button.pack(side=tk.LEFT, padx=2); self.selection_delete_button.pack(side=tk.LEFT, padx=2)
        else: self.selection_info_frame.pack_forget()

    def _rename_selected_item(self):
        if not self.selected_item: return
        new_name = simpledialog.askstring("Edit Name", "Enter new name:", initialvalue=self.selected_item['name'], parent=self)
        if new_name: self.selected_item['name'] = new_name; self._update_selection_info_panel(); self.redraw_canvas()

    def _delete_selected_item(self):
        if not self.selected_item: return
        def remove_from_list(item_list, item_to_remove):
            try: item_list.remove(item_to_remove); return True
            except ValueError: return False
        item_type = self.selected_item.get('type')
        found_and_removed = False
        if item_type == 'settlement': found_and_removed = remove_from_list(self.layers['settlements'], self.selected_item)
        elif item_type == 'placemark': found_and_removed = remove_from_list(self.layers['placemarks'], self.selected_item)
        else:
            for key in ['peaks', 'areas', 'ranges']:
                if key in self.layers['natural_features'] and remove_from_list(self.layers['natural_features'][key], self.selected_item):
                    found_and_removed = True; break
        if found_and_removed: self._deselect_item(); self.redraw_canvas()

    def _select_item(self, item): self.selected_item = item; self._update_selection_info_panel(); self.redraw_canvas()
    def _deselect_item(self): self.selected_item = None; self._update_selection_info_panel(); self.redraw_canvas()

    def _find_closest_item_to_click(self, event):
        if not self.generator: return None
        canvas_x, canvas_y, all_items = event.x, event.y, []
        for item_list in [self.layers.get('settlements', []), self.layers.get('placemarks', [])]: all_items.extend(item_list)
        if self.layers.get('natural_features'):
            for key in ['peaks', 'areas', 'ranges']: all_items.extend(self.layers['natural_features'].get(key, []))
        if not all_items: return None
        min_dist_sq, closest_item = float('inf'), None
        for item in all_items:
            item_x = item.get('x', item.get('center', [0,0])[0]); item_y = item.get('y', item.get('center', [0,0])[1])
            item_canvas_x, item_canvas_y = self.image_to_canvas_coords(item_x, item_y)
            if item_canvas_x is not None:
                dist_sq = (canvas_x - item_canvas_x)**2 + (canvas_y - item_canvas_y)**2
                if dist_sq < 20**2 and dist_sq < min_dist_sq: min_dist_sq, closest_item = dist_sq, item
        return closest_item

    def ease_in_out_cubic(self, x): return 4*x*x*x if x < 0.5 else 1 - pow(-2*x+2, 3)/2

    def _create_entry_widget(self, label_text, var, row, include_random_button=False, master=None):
        if master is None: master = self.controls_frame
        container = ttk.Frame(master); ttk.Label(container, text=label_text).pack(side=tk.LEFT, padx=(0,5)); ttk.Entry(container, textvariable=var, width=10).pack(side=tk.LEFT)
        if include_random_button: ttk.Button(container, text="🎲", width=3, command=lambda v=var: v.set(random.randint(0,100000))).pack(side=tk.LEFT, padx=(5,0))
        container.grid(row=row, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
        
    def _create_slider_widget(self, label_text, var, from_, to, row, master=None):
        if master is None: master = self.controls_frame
        container = ttk.Frame(master)
        if label_text: ttk.Label(container, text=label_text).grid(row=0, column=0, sticky='w', padx=5)
        is_double_var = isinstance(var, tk.DoubleVar)
        command_func = (lambda val, v=var: v.set(float(val))) if is_double_var else (lambda val, v=var: v.set(int(float(val))))
        scale = ttk.Scale(container, from_=from_, to=to, orient='horizontal', variable=var, command=command_func)
        scale.grid(row=1, column=0, sticky='ew', padx=5); ttk.Entry(container, textvariable=var, width=7).grid(row=1, column=1, sticky='w', padx=5)
        container.grid(row=row, column=0, columnspan=2, sticky='ew'); container.grid_columnconfigure(0, weight=1)

    def on_style_change(self):
        if not self.generator: return
        self.palette_combobox.set('Biome' if self.params['map_style'].get() == 'Biome' else 'Default')
        self.apply_predefined_palette(None)

    def on_projection_change(self):
        if self.params['projection'].get() == 'Orthographic': self.rotation_frame.grid()
        else: self.rotation_frame.grid_remove()
        self.redraw_canvas()

    def _on_canvas_press(self, event):
        if self.edit_mode.get():
            item = self._find_closest_item_to_click(event)
            if item: self._select_item(item); self.is_dragging = True; self.drag_start_pos = (event.x, event.y); self.canvas.config(cursor="fleur")
            else: self._deselect_item()
        elif self.placing_feature_info: self._add_placemark_at_click(event)
        else: self._on_pan_start(event)
    
    def _add_placemark_at_click(self, event):
        if not self.generator or not self.placing_feature_info: return
        if self.placing_feature_info.get('type') == 'PERSPECTIVE_VIEW':
            img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
            if img_x is not None:
                if self.perspective_viewer_instance and self.perspective_viewer_instance.winfo_exists(): self.perspective_viewer_instance.on_close()
                self.perspective_viewer_instance = PerspectiveViewer(self, (img_x, img_y))
            self.placing_feature_info = None; self.canvas.config(cursor=""); self.status_label.config(text="Ready")
            return
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        if img_x is None: return
        new_feature = {'name': self.placing_feature_info['name'], 'user_placed': True, 'type': 'placemark', 'x': img_x, 'y': img_y}
        self.layers['placemarks'].append(new_feature)
        self.placing_feature_info = None; self.canvas.config(cursor=""); self.status_label.config(text="Ready"); self.redraw_canvas()

    def _on_map_hover(self, event):
        if self.placing_feature_info or not self.generator or self.generator.color_map is None:
            self.tooltip.hide()
            return
            
        map_x, map_y = self.canvas_to_image_coords(event.x, event.y)
        if map_x is not None and 0 <= map_x < self.generator.x_range and 0 <= map_y < self.generator.y_range:
            elevation = self.generator.world_map.T[map_y, map_x]
            color_index = self.generator.color_map.T[map_y, map_x]
            
            display_text = None
            
            # Check for named water body first
            if not self.generator.land_mask.T[map_y, map_x] and self.generator.labeled_water is not None:
                water_label = self.generator.labeled_water.T[map_y, map_x]
                if water_label in self.feature_name_lookup:
                    display_text = self.feature_name_lookup[water_label]

            # If not a named water body, get biome name
            if display_text is None:
                display_text = self.get_biome_name_from_index(color_index)

            # Handle ice overlay text
            if display_text and display_text.lower() == "polar ice" and self.generator.color_map_before_ice is not None:
                underlying_name = self.get_biome_name_from_index(self.generator.color_map_before_ice.T[map_y, map_x])
                if underlying_name and underlying_name.lower() != "polar ice":
                     display_text = f"Polar Ice (over {underlying_name})"

            if display_text:
                self.tooltip.show(f"{display_text}\nCoords: ({map_x}, {map_y})\nElevation: {elevation:.2f}", event.x_root, event.y_root)
            else:
                self.tooltip.hide()
        else:
            self.tooltip.hide()

    def _on_map_leave(self, event): self.tooltip.hide()
        
    def get_biome_name_from_index(self, index):
        if self.params['map_style'].get() == 'Biome':
            for name, props in BIOME_DEFINITIONS.items():
                if 'shades' in props:
                    if props['idx'] <= index < props['idx'] + props['shades']:
                        return name.replace('_', ' ').title()
                elif props['idx'] == index:
                    return name.replace('_', ' ').title()
        # Fallback for terrain or unknown
        if 1<=index<=7: return "Water"
        if 16<=index<=31: return "Land"
        return "Unknown"
        
    def set_ui_state(self, is_generating):
        state = tk.DISABLED if is_generating else tk.NORMAL
        widget_list = [self.generate_button, self.random_button, self.recolor_button, self.palette_button, self.save_button, self.save_layers_button, self.palette_combobox, self.run_sim_button, self.regen_borders_button] # <-- ADD BUTTON HERE
        for widget in widget_list:
            if widget in [self.generate_button, self.random_button]: widget.config(state=state)
            else: widget.config(state=tk.NORMAL if not is_generating and self.generator else tk.DISABLED)

    def randomize_and_generate(self):
        self.params['seed'].set(random.randint(0,100000)); self.params['ice_seed'].set(random.randint(0,100000)); self.params['moisture_seed'].set(random.randint(0,100000))
        self.start_generation()

    def start_generation(self):
        self.set_ui_state(is_generating=True)
        self.frame_viewer_frame.grid_remove()
        self.simulation_frames, self.current_frame_index = [], -1
        self.layers = {'settlements':[], 'placemarks':[], 'rivers': [], 'borders': [], 'natural_features': {'peaks':[], 'ranges':[], 'areas':[], 'bays':[]}} # <-- ADD 'borders': []
        self.zoom, self.view_offset = 1.0, [0, 0]
        self.cached_globe_source_image = None
        params_dict = {key: var.get() for key, var in self.params.items() if key != 'world_size_preset'}
        size_preset = self.params['world_size_preset'].get()
        circumference_map = {'Kingdom (~1,600 km)':1600.0, 'Region (~6,400 km)':6400.0, 'Continent (~16,000 km)':16000.0}
        params_dict['world_circumference_km'] = circumference_map.get(size_preset, 6400.0)
        thread = threading.Thread(target=self.run_generation_in_thread, args=(params_dict,), daemon=True)
        thread.start()
        
    def run_generation_in_thread(self, params_dict):
        map_style = params_dict.get('map_style', 'Biome')
        self.palette = list(PREDEFINED_PALETTES['Biome' if map_style == 'Biome' else 'Default'])
        self.generator = FractalWorldGenerator(params_dict, self.palette, self.update_generation_progress)
        self.pil_image = self.generator.generate()
        self.after(0, self.finalize_generation)

    def finalize_generation(self):
        self.layers['settlements'].extend(self.generator.settlements)
        self.layers['rivers'].extend(self.generator.rivers)
        self.layers['borders'].extend(self.generator.borders)
        self.feature_name_lookup = {} # New lookup dict
        for key, value in self.generator.natural_features.items():
            if key in self.layers['natural_features']:
                self.layers['natural_features'][key].extend(value)
            else:
                self.layers['natural_features'][key] = value
            # Populate the lookup dict
            if key == 'areas':
                for feature in value:
                    if 'label' in feature:
                        self.feature_name_lookup[feature['label']] = feature['name']

        self.cached_globe_source_image = None
        self.redraw_canvas()
        self.set_ui_state(is_generating=False)
        self.status_label.config(text="Ready")

    def update_generation_progress(self, value, text): self.after(0, lambda: (self.progress.config(value=value), self.status_label.config(text=text)))

    def _create_text_with_outline(self, x, y, text, **kwargs):
        outline_kwargs = kwargs.copy(); outline_kwargs['fill'] = 'black'
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]: self.canvas.create_text(x+dx, y+dy, text=text, **outline_kwargs)
        self.canvas.create_text(x, y, text=text, **kwargs)

    def redraw_canvas(self, event=None):
        if not hasattr(self, 'generator') or not self.generator or not self.generator.pil_image: return
        self.tooltip.hide(); self.canvas.delete("all")
        if self.params['projection'].get() == 'Orthographic':
            source_array = np.array(self.generator.pil_image.convert('RGB'))
            canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if canvas_w <= 1 or canvas_h <= 1: return
            rot_y, rot_x = self.params['rotation_y'].get(), self.params['rotation_x'].get()
            new_img_array = render_globe(source_array, canvas_w, canvas_h, rot_y, rot_x, self.generator.x_range, self.generator.y_range)
            final_image = Image.fromarray(new_img_array).convert('RGBA')
            self.tk_image = ImageTk.PhotoImage(final_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image, tags="map_image")
        else:
            img_w, img_h = self.generator.pil_image.size; canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if self.zoom <= 0 or canvas_w <= 1 or canvas_h <= 1: return
            view_w, view_h = img_w/self.zoom, img_h/self.zoom
            x0 = self.view_offset[0]; y0 = max(0, min(self.view_offset[1], img_h - view_h if img_h > view_h else 0)); self.view_offset[1] = y0
            x0_wrapped = x0 % img_w; box1 = (x0_wrapped, y0, min(img_w, x0_wrapped + view_w), y0 + view_h)
            crop1 = self.generator.pil_image.crop(box1)
            stitched_image = Image.new('RGB', (int(view_w), int(view_h))); stitched_image.paste(crop1, (0, 0))
            if x0_wrapped + view_w > img_w:
                rem_w = (x0_wrapped + view_w) - img_w; box2 = (0, y0, rem_w, y0 + view_h)
                crop2 = self.generator.pil_image.crop(box2); stitched_image.paste(crop2, (int(crop1.width), 0))
            img_aspect, canvas_aspect = (view_w/view_h if view_h>0 else 1), (canvas_w/canvas_h if canvas_h>0 else 1)
            new_w, new_h = (canvas_w, int(canvas_w/img_aspect)) if img_aspect > canvas_aspect else (int(canvas_h*img_aspect), canvas_h)
            if new_w > 0 and new_h > 0:
                resized_image = stitched_image.resize((new_w, new_h), Image.Resampling.NEAREST)
                self.tk_image = ImageTk.PhotoImage(resized_image)
                px, py = (canvas_w-new_w)//2, (canvas_h-new_h)//2
                self.canvas.create_image(px, py, anchor=tk.NW, image=self.tk_image, tags="map_image")
                self.canvas_render_details = {'paste_x':px, 'paste_y':py, 'render_w':new_w, 'render_h':new_h}
                self._draw_layers_on_canvas_2d()
                self._draw_scale_bar()
            
    def _draw_scale_bar(self):
        self.canvas.delete("scale_bar")
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w <= 1 or not self.generator or self.zoom <= 0: return
        img_h = self.generator.y_range
        center_y_px = self.view_offset[1] + (img_h / self.zoom) / 2
        center_lat_rad = (center_y_px / img_h - 0.5) * math.pi
        if math.cos(center_lat_rad) <= 0: return
        km_per_px = (self.world_circumference_km * math.cos(center_lat_rad)) / (self.zoom * canvas_w)
        if km_per_px <= 0: return
        target_dist_km = (canvas_w / 5) * km_per_px
        if target_dist_km <= 0: return
        magnitude = 10**math.floor(math.log10(target_dist_km))
        residual = target_dist_km / magnitude
        if residual < 1.5: nice_dist = 1 * magnitude
        elif residual < 3.5: nice_dist = 2 * magnitude
        elif residual < 7.5: nice_dist = 5 * magnitude
        else: nice_dist = 10 * magnitude
        if nice_dist <= 0: return
        bar_width_px = nice_dist / km_per_px
        x, y = 20, canvas_h - 20
        self._create_text_with_outline(x+bar_width_px/2, y-10, text=f"{int(nice_dist)} km", fill="white", tags="scale_bar", anchor="s")
        self.canvas.create_line(x, y-5, x, y+5, fill="white", tags="scale_bar", width=2)
        self.canvas.create_line(x, y, x+bar_width_px, y, fill="white", tags="scale_bar", width=2)
        self.canvas.create_line(x+bar_width_px, y-5, x+bar_width_px, y+5, fill="white", tags="scale_bar", width=2)

    def _draw_layers_on_canvas_2d(self):
        self.canvas.delete("overlay")
        self._draw_borders_on_canvas_2d()
        self._draw_rivers_on_canvas_2d()
        self._draw_features_on_canvas_2d()
        self._draw_settlements_on_canvas_2d()
        self._draw_placemarks_on_canvas_2d()
        self._draw_selection_highlight_on_canvas_2d()

    def _draw_rivers_on_canvas_2d(self):
        if not self.layer_visibility['Rivers'].get() or 'rivers' not in self.layers: return
        river_color = "#6699FF" # A light blue for rivers
        for river in self.layers.get('rivers', []):
            if len(river['path']) < 2: continue
            
            # Convert all points at once
            path_points = [self.image_to_canvas_coords(p[0], p[1]) for p in river['path']]
            
            # Filter out points that are off-canvas
            valid_points = [p for p in path_points if p[0] is not None]
            
            if len(valid_points) > 1:
                # Flatten the list for create_line
                flat_points = [coord for point in valid_points for coord in point]
                width = min(5, 1 + math.log(river['flow'] + 1))
                self.canvas.create_line(*flat_points, fill=river_color, width=width, tags="overlay", capstyle=tk.ROUND, joinstyle=tk.ROUND)

    def _draw_features_on_canvas_2d(self):
        if not self.layer_visibility['Features'].get() or 'natural_features' not in self.layers: return
        for area in self.layers['natural_features'].get('areas',[]):
            cx,cy=self.image_to_canvas_coords(*area['center']);
            if cx is not None: self._create_text_with_outline(cx,cy,text=area['name'],font=("Arial",14,"italic"),fill="#DDEEFF",anchor="c",tags="overlay")
        for r in self.layers['natural_features'].get('ranges',[]):
            cx,cy=self.image_to_canvas_coords(*r['center']);
            if cx is not None: self._create_text_with_outline(cx,cy,text=r['name'],font=("Arial",12,"bold"),fill="#D2B48C",anchor="c",tags="overlay")
        for p in self.layers['natural_features'].get('peaks',[]):
            px,py=self.image_to_canvas_coords(p['x'],p['y']);
            if px is not None: self._create_text_with_outline(px,py-8,text=p['name'],font=("Arial",9),fill="white",anchor="s",tags="overlay"); self.canvas.create_text(px,py,text="▲",font=("Arial",12,"bold"),fill="white",tags="overlay"); self.canvas.create_text(px,py,text="▲",font=("Arial",12),fill="black",tags="overlay")

    def _draw_settlements_on_canvas_2d(self):
        if not self.layer_visibility['Settlements'].get() or 'settlements' not in self.layers: return
        for s in self.layers['settlements']:
            cx,cy = self.image_to_canvas_coords(s['x'],s['y']);
            if cx is not None: self.canvas.create_oval(cx-3,cy-3,cx+3,cy+3,fill='white',outline='black',tags="overlay"); self._create_text_with_outline(cx+5,cy,text=s['name'],anchor='w',font=("Arial",9),fill="white",tags="overlay")

    def _draw_placemarks_on_canvas_2d(self):
        if not self.layer_visibility['Placemarks'].get() or 'placemarks' not in self.layers: return
        for pm in self.layers['placemarks']:
            cx,cy = self.image_to_canvas_coords(pm['x'],pm['y']);
            if cx is not None: self.canvas.create_oval(cx-4,cy-4,cx+4,cy+4,fill='red',outline='black',tags="overlay"); self._create_text_with_outline(cx+6,cy,text=pm['name'],anchor='w',font=("Arial",10),fill="white",tags="overlay")

    def _draw_selection_highlight_on_canvas_2d(self):
        if not self.selected_item: return
        item_x = self.selected_item.get('x', self.selected_item.get('center', [0,0])[0]); item_y = self.selected_item.get('y', self.selected_item.get('center', [0,0])[1])
        cx,cy = self.image_to_canvas_coords(item_x,item_y)
        if cx is not None: self.canvas.create_oval(cx-15,cy-15,cx+15,cy+15,outline="yellow",width=3,tags="overlay")
    
    def image_to_canvas_coords(self, img_x, img_y):
        if not hasattr(self, 'generator') or not self.generator.pil_image: return None, None
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1: return None, None
        if self.params['projection'].get() == 'Orthographic': return map_coords_to_canvas(img_x, img_y, canvas_w, canvas_h, self.params['rotation_y'].get(), self.params['rotation_x'].get(), self.generator.x_range, self.generator.y_range)
        else:
            render_details = getattr(self, 'canvas_render_details', None)
            if not render_details: return None, None
            img_w, img_h = self.generator.pil_image.size
            if self.zoom <= 0 or img_w <= 0 or img_h <= 0: return None, None
            view_w, view_h = img_w/self.zoom, img_h/self.zoom
            dx = img_x - self.view_offset[0]
            if abs(dx) > img_w / 2: dx -= np.sign(dx) * img_w
            dy = img_y - self.view_offset[1]
            relative_x, relative_y = (dx/view_w)*render_details['render_w'] if view_w>0 else 0, (dy/view_h)*render_details['render_h'] if view_h>0 else 0
            return relative_x + render_details['paste_x'], relative_y + render_details['paste_y']
        
    def canvas_to_image_coords(self, canvas_x, canvas_y):
        if not hasattr(self, 'generator') or not self.generator.pil_image: return None, None
        if self.params['projection'].get() == 'Orthographic': return canvas_coords_to_map(canvas_x, canvas_y, self.canvas.winfo_width(), self.canvas.winfo_height(), self.params['rotation_y'].get(), self.params['rotation_x'].get(), self.generator.x_range, self.generator.y_range)
        else:
            render_details = getattr(self, 'canvas_render_details', None)
            if not render_details or render_details['render_w'] <= 0 or render_details['render_h'] <= 0: return None, None
            if not (render_details['paste_x'] <= canvas_x < render_details['paste_x'] + render_details['render_w'] and render_details['paste_y'] <= canvas_y < render_details['paste_y'] + render_details['render_h']): return None, None
            relative_x, relative_y = canvas_x - render_details['paste_x'], canvas_y - render_details['paste_y']
            percent_x, percent_y = relative_x / render_details['render_w'], relative_y / render_details['render_h']
            img_w, img_h = self.generator.pil_image.size; view_w, view_h = img_w / self.zoom, img_h / self.zoom
            img_x, img_y = self.view_offset[0] + (percent_x * view_w), self.view_offset[1] + (percent_y * view_h)
            return int(img_x % img_w), int(img_y)

    def _on_zoom(self, event):
        if not self.generator or not self.generator.pil_image or self.params['projection'].get() == 'Orthographic': return
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1: return
        img_x_before, img_y_before = self.canvas_to_image_coords(event.x, event.y)
        if img_x_before is None: return
        factor = 1.1 if event.delta > 0 else 0.9
        new_zoom = max(0.1, min(self.zoom * factor, 50))
        img_w, img_h = self.generator.pil_image.size
        view_w_before, view_h_before = (img_w / self.zoom if self.zoom > 0 else 0), (img_h / self.zoom if self.zoom > 0 else 0)
        if view_w_before <= 0 or view_h_before <= 0: return
        view_w_after, view_h_after = img_w / new_zoom, img_h / new_zoom
        self.view_offset[0] += (img_x_before - self.view_offset[0]) * (1 - view_w_after / view_w_before)
        self.view_offset[1] += (img_y_before - self.view_offset[1]) * (1 - view_h_after / view_h_before)
        self.zoom = new_zoom; self.redraw_canvas()

    def _on_pan_start(self, event):
        if self.placing_feature_info: return
        self.pan_start_pos = (event.x, event.y); self.canvas.config(cursor="fleur")

    def _on_pan_end(self, event):
        self.pan_start_pos = None; self.is_dragging = False; self.drag_start_pos = None
        self.canvas.config(cursor="hand2" if self.edit_mode.get() else "")

    def _on_pan_move(self, event):
        if self.edit_mode.get() and self.is_dragging and self.selected_item:
            start_img_x, start_img_y = self.canvas_to_image_coords(*self.drag_start_pos)
            end_img_x, end_img_y = self.canvas_to_image_coords(event.x, event.y)
            if start_img_x is None or end_img_x is None: return
            delta_img_x, delta_img_y = end_img_x-start_img_x, end_img_y-start_img_y
            img_w, _ = self.generator.pil_image.size
            if 'x' in self.selected_item: self.selected_item['x'] = (self.selected_item['x']+delta_img_x)%img_w; self.selected_item['y'] += delta_img_y
            elif 'center' in self.selected_item: cx,cy = self.selected_item['center']; self.selected_item['center'] = ((cx+delta_img_x)%img_w, cy+delta_img_y)
            self.drag_start_pos = (event.x, event.y); self.redraw_canvas()
        elif not self.edit_mode.get() and self.pan_start_pos:
            dx, dy = event.x - self.pan_start_pos[0], event.y - self.pan_start_pos[1]
            if self.params['projection'].get() == 'Orthographic':
                self.params['rotation_y'].set(self.params['rotation_y'].get() - dx*0.3); self.params['rotation_x'].set(max(-90, min(90, self.params['rotation_x'].get() - dy*0.3)))
            else:
                if not self.generator or not self.generator.pil_image: return
                img_w, img_h = self.generator.pil_image.size; canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
                if canvas_w > 0 and canvas_h > 0 and self.zoom > 0:
                    px_per_canvas_x, px_per_canvas_y = (img_w/self.zoom)/canvas_w, (img_h/self.zoom)/canvas_h
                    self.view_offset[0] -= dx*px_per_canvas_x; self.view_offset[1] -= dy*px_per_canvas_y
            self.pan_start_pos = (event.x, event.y); self.redraw_canvas()
        
    def open_placemark_dialog(self):
        if not self.generator: tk.messagebox.showwarning("No World", "Please generate a world first."); return
        dialog = PlacemarkDialog(self)
        if dialog.result: self.placing_feature_info = dialog.result; self.canvas.config(cursor="crosshair"); self.status_label.config(text=f"Click to place '{self.placing_feature_info['name']}'...")

    def save_image(self):
        if not self.generator or not self.generator.pil_image: tk.messagebox.showwarning("No Image", "Please generate a world before saving."); return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")], title="Save Image As")
        if not file_path: return
        export_image = None
        if self.params['projection'].get() == 'Orthographic':
            export_size = 1600; source_array = np.array(self.generator.pil_image.convert('RGB'))
            rot_y, rot_x = self.params['rotation_y'].get(), self.params['rotation_x'].get()
            new_img_array = render_globe(source_array, export_size, export_size, rot_y, rot_x, self.generator.x_range, self.generator.y_range)
            export_image = Image.fromarray(new_img_array).convert('RGBA')
        else:
            export_image = self.generator.pil_image.copy().convert("RGBA"); draw = ImageDraw.Draw(export_image)
            try: font_l,font_m,font_s = ImageFont.truetype("arialbd.ttf", 16), ImageFont.truetype("arialbd.ttf", 11), ImageFont.truetype("arial.ttf", 10)
            except IOError: font_l,font_m,font_s = ImageFont.load_default(), ImageFont.load_default(), ImageFont.load_default()
            def draw_outlined_text(pos, text, font, fill, **kwargs):
                x, y = pos; stroke_fill, stroke_width = 'black', 1
                draw.text((x-stroke_width, y), text, font=font, fill=stroke_fill, **kwargs); draw.text((x+stroke_width, y), text, font=font, fill=stroke_fill, **kwargs)
                draw.text((x, y-stroke_width), text, font=font, fill=stroke_fill, **kwargs); draw.text((x, y+stroke_width), text, font=font, fill=stroke_fill, **kwargs)
                draw.text(pos, text, font=font, fill=fill, **kwargs)
            
            if self.layer_visibility['Rivers'].get():
                river_color = "#4a7fdd"
                for river in self.layers.get('rivers', []):
                    if len(river['path']) < 2: continue
                    width = min(5, 1 + math.log(river['flow'] + 1))
                    draw.line(river['path'], fill=river_color, width=int(width), joint='curve')

            if self.layer_visibility['Features'].get():
                for area in self.layers['natural_features'].get('areas',[]): draw_outlined_text((int(area['center'][0]), int(area['center'][1])), area['name'], font_l, "#DDEEFF", anchor="mm")
                for r in self.layers['natural_features'].get('ranges',[]): draw_outlined_text((int(r['center'][0]), int(r['center'][1])), r['name'], font_l, "#D2B48C", anchor="mm")
                for p in self.layers['natural_features'].get('peaks',[]): x,y=int(p['x']),int(p['y']); draw_outlined_text((x,y-8),p['name'],font_s,"white",anchor="ms"); draw.text((x,y),"▲",font=font_m,fill="black",anchor="mm")
            if self.layer_visibility['Settlements'].get():
                for s in self.layers['settlements']: x,y=int(s['x']),int(s['y']); draw.ellipse((x-3,y-3,x+3,y+3),fill='white',outline='black'); draw_outlined_text((x+5,y),s['name'],font_s,"white",anchor="lm")
            if self.layer_visibility['Placemarks'].get():
                for pm in self.layers['placemarks']: x,y=int(pm['x']),int(pm['y']); draw.ellipse((x-4,y-4,x+4,y+4),fill='red',outline='black'); draw_outlined_text((x+6,y),pm['name'],font_m,"white",anchor="lm")
        if export_image:
            try: export_image.save(file_path); tk.messagebox.showinfo("Success", f"Image saved to:\n{file_path}")
            except Exception as e: tk.messagebox.showerror("Save Error", f"Failed to save image:\n{e}")

    def save_layered_image(self):
        if not self.generator or not self.generator.pil_image: tk.messagebox.showwarning("No Image", "Please generate a world before saving."); return
        directory_path = filedialog.askdirectory(title="Select Directory to Save Layers")
        if not directory_path: return
        try:
            w, h = self.generator.pil_image.size
            try: font_l,font_m,font_s = ImageFont.truetype("arialbd.ttf", 16), ImageFont.truetype("arialbd.ttf", 11), ImageFont.truetype("arial.ttf", 10)
            except IOError: font_l,font_m,font_s = ImageFont.load_default(), ImageFont.load_default(), ImageFont.load_default()
            def draw_outlined_text(draw_context, pos, text, font, fill, **kwargs):
                x,y=pos; sf,sw='black',1; draw_context.text((x-sw,y),text,font=font,fill=sf,**kwargs); draw_context.text((x+sw,y),text,font=font,fill=sf,**kwargs); draw_context.text((x,y-sw),text,font=font,fill=sf,**kwargs); draw_context.text((x,y+sw),text,font=font,fill=sf,**kwargs); draw_context.text(pos,text,font=font,fill=fill,**kwargs)
            
            self.generator.pil_image.save(os.path.join(directory_path, "00_background.png"))

            if self.layer_visibility['Rivers'].get() and self.layers['rivers']:
                layer_img=Image.new("RGBA",(w,h),(0,0,0,0)); draw=ImageDraw.Draw(layer_img)
                river_color = "#4a7fdd"
                for river in self.layers.get('rivers', []):
                    if len(river['path']) < 2: continue
                    width = min(5, 1 + math.log(river['flow'] + 1))
                    draw.line(river['path'], fill=river_color, width=int(width), joint='curve')
                layer_img.save(os.path.join(directory_path, "01_rivers.png"))

            if self.layer_visibility['Features'].get():
                layer_defs = [('areas','02_areas.png',font_l,"#DDEEFF", "mm"), ('ranges','03_ranges.png',font_l,"#D2B48C", "mm"), ('peaks','04_peaks.png',font_s,"white", "ms")]
                for key, fname, font, color, anchor in layer_defs:
                    if not self.layers['natural_features'].get(key): continue
                    layer_img=Image.new("RGBA",(w,h),(0,0,0,0)); draw=ImageDraw.Draw(layer_img)
                    for item in self.layers['natural_features'][key]:
                        x,y = (int(item['center'][0]),int(item['center'][1])) if 'center' in item else (int(item['x']),int(item['y']))
                        if key=='peaks': draw_outlined_text(draw,(x,y-8),item['name'],font,color,anchor=anchor); draw.text((x,y),"▲",font=font_m,fill="black",anchor="mm")
                        else: draw_outlined_text(draw,(x,y),item['name'],font,color,anchor=anchor)
                    layer_img.save(os.path.join(directory_path, fname))
            if self.layer_visibility['Settlements'].get() and self.layers['settlements']:
                layer_img=Image.new("RGBA",(w,h),(0,0,0,0)); draw=ImageDraw.Draw(layer_img)
                for s in self.layers['settlements']: x,y=int(s['x']),int(s['y']); draw.ellipse((x-3,y-3,x+3,y+3),fill='white',outline='black'); draw_outlined_text(draw,(x+5,y),s['name'],font_s,"white",anchor="lm")
                layer_img.save(os.path.join(directory_path, "05_settlements.png"))
            if self.layer_visibility['Placemarks'].get() and self.layers['placemarks']:
                layer_img=Image.new("RGBA",(w,h),(0,0,0,0)); draw=ImageDraw.Draw(layer_img)
                for pm in self.layers['placemarks']: x,y=int(pm['x']),int(pm['y']); draw.ellipse((x-4,y-4,x+4,y+4),fill='red',outline='black'); draw_outlined_text(draw,(x+6,y),pm['name'],font_m,"white",anchor="lm")
                layer_img.save(os.path.join(directory_path, "06_placemarks.png"))
            tk.messagebox.showinfo("Success", f"Layers saved successfully to:\n{directory_path}")
        except Exception as e: tk.messagebox.showerror("Save Error", f"Failed to save layers:\n{e}")

    def save_preset(self):
        params_to_save = {key: var.get() for key, var in self.params.items()}; file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Preset File", "*.json")], title="Save Preset As")
        if not file_path: return
        try:
            with open(file_path, 'w', encoding='utf-8') as f: json.dump(params_to_save, f, indent=4)
        except Exception as e: tk.messagebox.showerror("Save Error", f"Failed to save preset:\n{e}")

    def load_preset(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON Preset File", "*.json")], title="Load Preset")
        if not file_path: return
        try:
            with open(file_path, 'r', encoding='utf-8') as f: loaded_params = json.load(f)
            for key, var in self.params.items():
                if key in loaded_params: var.set(loaded_params[key])
            # --- THIS LINE IS THE FIX ---
            self.layers = {'settlements': [], 'placemarks': [], 'rivers': [], 'borders': [], 'natural_features': {'peaks': [], 'ranges': [], 'areas': [], 'bays': []}}
            self.pil_image, self.generator = None, None
            self.canvas.delete("all"); self.status_label.config(text="Preset loaded. Click 'Regenerate World'.")
            self.on_projection_change(); self.on_style_change()
        except Exception as e: tk.messagebox.showerror("Load Error", f"Failed to load preset:\n{e}")

    def open_palette_editor(self):
        if not self.generator: tk.messagebox.showwarning("No World", "Please generate a world first."); return
        PaletteEditor(self, self.apply_palette_from_editor).grab_set()

    def apply_palette_from_editor(self, new_palette): self.palette=new_palette; self.recolor_map()
        
    def recolor_map(self):
        if not self.generator or self.generator.color_map is None: return
        self.set_ui_state(is_generating=True); thread=threading.Thread(target=self.run_recolor_in_thread, daemon=True); thread.start()
        
    def run_recolor_in_thread(self):
        self.update_generation_progress(10, "Recoloring...")
        self.generator.palette = self.palette; self.generator.pil_image = self.generator.create_image()
        self.update_generation_progress(80, "Applying new palette...")
        self.after(0, self.finalize_recolor)

    def finalize_recolor(self):
        self.cached_globe_source_image = None; self.redraw_canvas()
        self.set_ui_state(is_generating=False); self.status_label.config(text="Ready")
        
    def apply_predefined_palette(self, event=None):
        if not self.generator: return
        self.palette = list(PREDEFINED_PALETTES[self.palette_combobox.get()]); self.recolor_map()
        
    def start_age_simulation(self):
        if not self.generator: tk.messagebox.showwarning("No Map", "Please generate a map first."); return
        save_dir = filedialog.askdirectory(title="Select Folder to Save Animation Frames")
        if not save_dir: return
        self.set_ui_state(is_generating=True); self.frame_viewer_frame.grid_remove()
        self.simulation_frames, self.current_frame_index = [], -1
        sim_params = {'event': self.params['simulation_event'].get(), 'frames': self.params['simulation_frames'].get(), 'save_dir': save_dir, 'base_filename': os.path.basename(save_dir)}
        thread = threading.Thread(target=self.run_age_simulation_in_thread, args=(sim_params,), daemon=True)
        thread.start()

    def run_age_simulation_in_thread(self, sim_params):
        event, frames, save_dir, base_filename = sim_params['event'], sim_params['frames'], sim_params['save_dir'], sim_params['base_filename']
        start_water, start_ice = self.params['water'].get(), self.params['ice'].get()
        original_ice_seed, thaw_seed = self.params['ice_seed'].get(), self.params['thaw_ice_seed'].get()
        use_separate_thaw_seed = thaw_seed != original_ice_seed
        
        if event == 'Ice Age Cycle':
            peak_ice, peak_water, max_temp_offset = 90.0, start_water - 10, -0.15
            midpoint = frames / 2.0; switched_to_thaw_noise = False
            for i in range(frames):
                self.update_generation_progress(int((i+1)/frames*100), f"Rendering frame {i+1}/{frames}...")
                raw_progress = (i/midpoint) if i < midpoint else ((i-midpoint)/midpoint)
                eased_progress = self.ease_in_out_cubic(raw_progress)
                if i < midpoint:
                    current_ice, current_water, current_temp_offset = start_ice+(peak_ice-start_ice)*eased_progress, start_water+(peak_water-start_water)*eased_progress, max_temp_offset*eased_progress
                else:
                    if use_separate_thaw_seed and not switched_to_thaw_noise: self.generator.set_ice_seed(thaw_seed); switched_to_thaw_noise=True
                    current_ice, current_water, current_temp_offset = peak_ice-(peak_ice-start_ice)*eased_progress, peak_water-(peak_water-start_water)*eased_progress, max_temp_offset*(1.0-eased_progress)
                self.generator.finalize_map(current_water, current_ice, self.params['map_style'].get(), temp_offset=current_temp_offset)
                frame_image = self.generator.create_image()
                self.simulation_frames.append({'image':frame_image.copy(), 'color_map':self.generator.color_map.copy(), 'color_map_before_ice':self.generator.color_map_before_ice.copy()})
                self.generator.pil_image = frame_image
                self.after(0, self.redraw_canvas)
                frame_image.save(os.path.join(save_dir, f"{base_filename}_{i+1:03d}.png")); time.sleep(0.05)
        if use_separate_thaw_seed: self.generator.set_ice_seed(original_ice_seed)
        self.after(0, self.finalize_simulation)

    def finalize_simulation(self):
        if self.simulation_frames:
            self.current_frame_index = len(self.simulation_frames) - 1; self.display_simulation_frame(self.current_frame_index)
            self.frame_viewer_frame.grid(row=self.frame_viewer_row, columnspan=3, sticky='ew', padx=5, pady=5)
        self.set_ui_state(is_generating=False); self.status_label.config(text="Ready")

    def show_previous_frame(self):
        if self.simulation_frames and self.current_frame_index > 0: self.current_frame_index -= 1; self.display_simulation_frame(self.current_frame_index)
    def show_next_frame(self):
        if self.simulation_frames and self.current_frame_index < len(self.simulation_frames)-1: self.current_frame_index += 1; self.display_simulation_frame(self.current_frame_index)
    def display_simulation_frame(self, index):
        if not self.simulation_frames or not (0<=index<len(self.simulation_frames)): return
        frame_data = self.simulation_frames[index]
        self.generator.pil_image, self.generator.color_map, self.generator.color_map_before_ice = frame_data['image'], frame_data['color_map'], frame_data['color_map_before_ice']
        self.redraw_canvas()
        self.frame_label.config(text=f"{index+1}/{len(self.simulation_frames)}")
        self.prev_frame_button.config(state=tk.NORMAL if index>0 else tk.DISABLED)
        self.next_frame_button.config(state=tk.NORMAL if index<len(self.simulation_frames)-1 else tk.DISABLED)

    def draw_perspective_fov(self, camera_pos, angle_deg, fov_deg=72, length=50):
        self.canvas.delete("fov_lines")
        if self.params['projection'].get() != 'Equirectangular': return
        cam_img_x, cam_img_y = camera_pos
        cam_canvas_x, cam_canvas_y = self.image_to_canvas_coords(cam_img_x, cam_img_y)
        if cam_canvas_x is None: return
        angle_rad, fov_rad = math.radians(angle_deg), math.radians(fov_deg)
        left_angle, right_angle = angle_rad - fov_rad/2, angle_rad + fov_rad/2
        end_left_x, end_left_y = cam_img_x-math.sin(left_angle)*length, cam_img_y+math.cos(left_angle)*length
        end_right_x, end_right_y = cam_img_x-math.sin(right_angle)*length, cam_img_y+math.cos(right_angle)*length
        end_left_canvas_x, end_left_canvas_y = self.image_to_canvas_coords(end_left_x, end_left_y)
        end_right_canvas_x, end_right_canvas_y = self.image_to_canvas_coords(end_right_x, end_right_y)
        if end_left_canvas_x is not None: self.canvas.create_line(cam_canvas_x, cam_canvas_y, end_left_canvas_x, end_left_canvas_y, fill="yellow", width=2, tags="fov_lines")
        if end_right_canvas_x is not None: self.canvas.create_line(cam_canvas_x, cam_canvas_y, end_right_canvas_x, end_right_canvas_y, fill="yellow", width=2, tags="fov_lines")
        self.canvas.create_oval(cam_canvas_x-4, cam_canvas_y-4, cam_canvas_x+4, cam_canvas_y+4, fill="yellow", outline="black", tags="fov_lines")

    def clear_perspective_fov(self): self.canvas.delete("fov_lines")

    def set_perspective_view(self):
        if not self.generator or not self.generator.pil_image: tk.messagebox.showwarning("No Map", "Please generate a map first."); return
        if self.perspective_viewer_instance and self.perspective_viewer_instance.winfo_exists(): self.perspective_viewer_instance.lift(); return
        self.placing_feature_info = {'type': 'PERSPECTIVE_VIEW'}; self.canvas.config(cursor="crosshair"); self.status_label.config(text="Click to set camera position...")

    def regenerate_borders(self):
        if not self.generator or not self.generator.settlements:
            tk.messagebox.showwarning("Cannot Generate Borders", "Please generate a world with settlements first.", parent=self)
            return

        border_params = {
            'num_kingdoms': self.params['num_kingdoms'].get(),
            'mountain_cost': self.params['mountain_cost'].get(),
            'hill_cost': self.params['hill_cost'].get(),
        }

        self.set_ui_state(is_generating=True)
        self.status_label.config(text="Regenerating political borders...")
        thread = threading.Thread(target=self.run_border_generation_in_thread, args=(border_params,), daemon=True)
        thread.start()

    def run_border_generation_in_thread(self, border_params):
        """Runs the border generation in a separate thread to avoid freezing the UI."""
        self.generator.generate_borders(border_params)
        self.after(0, self.finalize_border_generation)

    def finalize_border_generation(self):
        """Updates the UI after the border generation thread is complete."""
        self.layers['borders'] = list(self.generator.borders)
        self.set_ui_state(is_generating=False)
        self.status_label.config(text="Ready")
        self.redraw_canvas()

if __name__ == "__main__":
    app = App()
    app.mainloop()
