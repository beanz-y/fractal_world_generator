# beanz-y/fractal_world_generator/fractal_world_generator-28f75751b57dacf83432892d2293f1e3754a3ba6/viewers.py

import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
import math
import threading
from queue import Queue
import numba

@numba.jit(nopython=True)
def _numba_scalar_clip(value, min_val, max_val):
    if value < min_val: return min_val
    elif value > max_val: return max_val
    else: return value

@numba.jit(nopython=True, fastmath=True)
def _numba_render_perspective(height_map, color_map, palette, camera_pos, view_angle_rad, fov, width, height, max_distance, camera_altitude_offset, vertical_exaggeration):
    map_h, map_w = height_map.shape
    cam_x, cam_y = camera_pos
    safe_cam_y = _numba_scalar_clip(int(cam_y), 0, map_h - 1)
    safe_cam_x = _numba_scalar_clip(int(cam_x), 0, map_w - 1)
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
            r, g, b = [int(sky*(1-interp) + fog*interp) for sky, fog in zip(sky_color, fog_color)]
            image_buffer[y, :, 0], image_buffer[y, :, 1], image_buffer[y, :, 2] = r, g, b

    water_level = np.percentile(height_map.ravel(), 60.0)

    for i in range(width):
        y_buffer = height
        angle = view_angle_rad - fov/2 + fov * i/width
        sin_angle, cos_angle = -math.sin(angle), math.cos(angle)
        for z in range(1, int(max_distance)):
            map_x, map_y = int(cam_x + sin_angle*z), int(cam_y + cos_angle*z)
            map_x_wrapped = map_x % map_w
            if not (0 <= map_y < map_h): continue
            terrain_height = height_map[map_y, map_x_wrapped]
            color_index = color_map[map_y, map_x_wrapped]
            is_water = 1 <= color_index <= 7 and terrain_height < water_level
            display_height = (water_level if is_water else terrain_height) * vertical_exaggeration
            base_color = palette[color_index]
            projected_height = int((cam_height - display_height)/z * 240 + horizon)
            if projected_height >= y_buffer: continue
            fog_factor = min(1.0, (z / max_distance)**1.5)
            r,g,b = [int(base*(1-fog_factor) + fog*fog_factor) for base, fog in zip(base_color, fog_color)]
            draw_start_y, draw_end_y = _numba_scalar_clip(projected_height,0,height), _numba_scalar_clip(y_buffer,0,height)
            for y_draw in range(draw_start_y, draw_end_y):
                image_buffer[y_draw, i, 0], image_buffer[y_draw, i, 1], image_buffer[y_draw, i, 2] = r,g,b
            y_buffer = projected_height
            if y_buffer <= 0: break
    return image_buffer

class PerspectiveViewer(tk.Toplevel):
    """A Toplevel window for rendering a 3D perspective view of the map."""
    def __init__(self, parent, camera_pos):
        super().__init__(parent)
        self.parent, self.generator = parent, parent.generator
        self.camera_pos, self.view_angle = camera_pos, 180.0
        self.image, self.tk_image = None, None
        self.rendering_thread, self._is_rendering = None, threading.Event()
        self.camera_altitude = tk.DoubleVar(value=40.0)
        self.vertical_exaggeration = tk.DoubleVar(value=2.5)

        self.title(f"Perspective View from {self.camera_pos}")
        self.geometry("800x680")
        self.render_queue = Queue()

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(main_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.display_image(rerender=False))

        controls_frame = ttk.Frame(self)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        self.pan_left_button = ttk.Button(controls_frame, text="Pan Left", command=lambda: self.pan(-15.0)); self.pan_left_button.pack(side=tk.LEFT, padx=5)
        self.pan_right_button = ttk.Button(controls_frame, text="Pan Right", command=lambda: self.pan(15.0)); self.pan_right_button.pack(side=tk.LEFT, padx=5)
        self.save_button = ttk.Button(controls_frame, text="Save Image...", command=self.save_image, state=tk.DISABLED); self.save_button.pack(side=tk.RIGHT, padx=5)

        sliders_frame = ttk.Frame(self); sliders_frame.pack(fill=tk.X, padx=10, pady=(5, 10)); sliders_frame.columnconfigure(1, weight=1)
        ttk.Label(sliders_frame, text="Altitude:").grid(row=0, column=0, sticky='w', padx=5)
        ttk.Scale(sliders_frame, from_=5, to=500, orient='horizontal', variable=self.camera_altitude, command=self.on_slider_change).grid(row=0, column=1, sticky='ew', padx=5)
        ttk.Label(sliders_frame, text="Exaggeration:").grid(row=1, column=0, sticky='w', padx=5)
        ttk.Scale(sliders_frame, from_=1.0, to=15.0, orient='horizontal', variable=self.vertical_exaggeration, command=self.on_slider_change).grid(row=1, column=1, sticky='ew', padx=5)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_parent_fov()
        self.start_render_thread()
        self.check_render_queue()

    def on_slider_change(self, value):
        if not self._is_rendering.is_set(): self.start_render_thread()
    def update_parent_fov(self): self.parent.draw_perspective_fov(self.camera_pos, self.view_angle)

    def on_close(self):
        self._is_rendering.set()
        if self.rendering_thread and self.rendering_thread.is_alive(): self.rendering_thread.join(timeout=0.1)
        self.parent.clear_perspective_fov()
        self.parent.perspective_viewer_instance = None
        self.destroy()

    def pan(self, angle_change):
        if self._is_rendering.is_set(): return
        self.view_angle += angle_change
        self.update_parent_fov(); self.start_render_thread()

    def start_render_thread(self):
        if self._is_rendering.is_set(): return
        self._is_rendering.set()
        self.save_button.config(state=tk.DISABLED); self.pan_left_button.config(state=tk.DISABLED); self.pan_right_button.config(state=tk.DISABLED)
        self.canvas.delete("render_text")
        self.canvas.create_text(self.canvas.winfo_width()/2, self.canvas.winfo_height()/2, text="Rendering...", fill="yellow", font=("Arial", 24, "bold"), tags="render_text")
        self.rendering_thread = threading.Thread(target=self.render_view_threaded, daemon=True)
        self.rendering_thread.start()

    def render_view_threaded(self):
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 10 or height < 10: width, height = 800, 600
        render_w, render_h = width * 2, height * 2
        fov, max_dist = math.pi/2.5, self.generator.x_range/1.5
        palette_np = np.array(self.generator.palette, dtype=np.uint8)
        altitude, exaggeration = self.camera_altitude.get(), self.vertical_exaggeration.get()
        image_data = _numba_render_perspective(self.generator.world_map.T, self.generator.color_map.T, palette_np, self.camera_pos, math.radians(self.view_angle), fov, render_w, render_h, max_dist, altitude, exaggeration)
        if not self.winfo_exists(): return
        final_image = Image.fromarray(image_data, 'RGB').resize((width, height), Image.Resampling.LANCZOS)
        self.render_queue.put(final_image)

    def check_render_queue(self):
        try:
            new_image = self.render_queue.get_nowait()
            self.image = new_image
            self.display_image(rerender=True)
            self._is_rendering.clear()
            self.save_button.config(state=tk.NORMAL); self.pan_left_button.config(state=tk.NORMAL); self.pan_right_button.config(state=tk.NORMAL)
        except Exception: pass
        finally:
            if self.winfo_exists(): self.after(100, self.check_render_queue)

    def display_image(self, rerender=True):
        if self.image:
            if rerender: self.canvas.delete("all")
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def save_image(self):
        if self.image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")], title="Save Perspective View As")
            if file_path: self.image.save(file_path)