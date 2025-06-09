import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
import random
import math
import threading
from collections import deque

class FractalWorldGenerator:
    """
    This class encapsulates the logic for generating the fractal world map,
    ported from the original C++ code.
    """

    def __init__(self, params, progress_callback=None):
        """
        Initializes the generator with user-defined parameters.

        Args:
            params (dict): A dictionary containing all generation parameters.
            progress_callback (func): A function to call to update a progress bar.
        """
        self.x_range = params['width']
        self.y_range = params['height']
        self.num_faults = params['faults']
        self.percent_water = params['water']
        self.percent_ice = params['ice']
        self.seed = params['seed']
        self.progress_callback = progress_callback
        
        # Initialize the random number generator
        random.seed(self.seed)
        
        # Define the color palette (ported from C++ code)
        self.palette = [
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
        ]

        # Use a very small number to represent INT_MIN for initialization
        self.INT_MIN_PLACEHOLDER = -2**31
        self.world_map = None

    def _add_fault(self):
        """
        Applies a single random great-circle fault to the map.
        This is a direct port of the GenerateWorldMap C++ function.
        """
        flag1 = random.randint(0, 1)

        # Create a random great circle by rotating an equator
        alpha = (random.random() - 0.5) * np.pi  # Rotate around x-axis
        beta = (random.random() - 0.5) * np.pi   # Rotate around y-axis

        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)

        # Avoid math domain error for acos
        cos_alpha_cos_beta = np.clip(cos_alpha * cos_beta, -1.0, 1.0)
        tan_b = np.tan(np.arccos(cos_alpha_cos_beta))
        
        xsi = int(self.x_range / 2.0 - (self.x_range / np.pi) * beta)

        for phi in range(self.x_range // 2):
            # Calculate theta (latitude) for the great circle at this phi (longitude)
            sin_val = self.sin_iter_phi[xsi - phi + self.x_range]
            theta = int((self.y_range_div_pi * np.arctan(sin_val * tan_b)) + self.y_range_div_2)
            
            # Clamp theta to be within the map bounds
            theta = max(0, min(self.y_range - 1, theta))
            
            idx = (phi, theta)
            current_val = self.world_map[idx]
            
            if flag1: # Rise northern hemisphere, lower southern
                delta = -1
            else: # Rise southern hemisphere, lower northern
                delta = 1
            
            if current_val != self.INT_MIN_PLACEHOLDER:
                self.world_map[idx] += delta
            else:
                self.world_map[idx] = delta
                
    def _flood_fill(self, start_x, start_y, old_color, max_fill_percent):
        """
        Iterative flood fill to create ice caps.
        Starts at a given point and fills connected areas with ice colors.
        
        Args:
            start_x, start_y (int): The starting coordinates for the fill.
            old_color (int): The color index to be replaced.
            max_fill_percent (float): The percentage of the map to fill with ice.

        Returns:
            int: The number of pixels filled.
        """
        max_pixels_to_fill = (self.x_range * self.y_range) * (max_fill_percent / 100.0)
        
        if self.world_map[start_x, start_y] != old_color:
            return 0
            
        filled_pixels = 0
        q = deque([(start_x, start_y)])

        while q:
            x, y = q.popleft()

            if self.world_map[x, y] != old_color:
                continue

            # Apply ice color
            if self.world_map[x, y] < 16: # Water
                self.world_map[x, y] = 32 # Sea ice
            else: # Land
                self.world_map[x, y] += 17 # Land ice/snow

            filled_pixels += 1
            if filled_pixels >= max_pixels_to_fill:
                return filled_pixels

            # Check neighbors (4-connectivity)
            # West (with wraparound)
            nx, ny = (x - 1 + self.x_range) % self.x_range, y
            if self.world_map[nx, ny] == old_color: q.append((nx, ny))
            
            # East (with wraparound)
            nx, ny = (x + 1) % self.x_range, y
            if self.world_map[nx, ny] == old_color: q.append((nx, ny))

            # North
            if y > 0:
                nx, ny = x, y - 1
                if self.world_map[nx, ny] == old_color: q.append((nx, ny))

            # South
            if y < self.y_range - 1:
                nx, ny = x, y + 1
                if self.world_map[nx, ny] == old_color: q.append((nx, ny))

        return filled_pixels

    def generate(self):
        """
        The main generation pipeline.
        
        Returns:
            PIL.Image: The generated world map as an image.
        """
        # 1. Initialization
        self.world_map = np.full((self.x_range, self.y_range), self.INT_MIN_PLACEHOLDER, dtype=np.int32)
        # Initialize top row to 0, which might be changed by faults.
        self.world_map[:, 0] = 0
        
        # Precompute constants and lookup tables
        self.y_range_div_2 = self.y_range / 2.0
        self.y_range_div_pi = self.y_range / np.pi
        self.sin_iter_phi = np.sin(np.arange(2 * self.x_range) * 2 * np.pi / self.x_range)
        
        # 2. Add Faults (Main fractal generation loop)
        for i in range(self.num_faults):
            self._add_fault()
            if self.progress_callback:
                self.progress_callback(i + 1)

        # 3. Post-processing
        # Copy the calculated half of the map to the other half with symmetry
        for j in range(self.x_range // 2):
            for i in range(1, self.y_range):
                self.world_map[j + self.x_range // 2, self.y_range - i] = self.world_map[j, i]

        # Reconstruct the real heightmap from the fault lines
        # This is a cumulative sum down each column, but only where values are not the placeholder.
        temp_map = self.world_map.copy()
        temp_map[temp_map == self.INT_MIN_PLACEHOLDER] = 0
        self.world_map = np.cumsum(temp_map, axis=1)

        # 4. Determine Sea Level
        min_z = np.min(self.world_map)
        max_z = np.max(self.world_map)
        
        if max_z == min_z: max_z = min_z + 1 # Avoid division by zero
        
        # Create a histogram of height values to find the water threshold
        hist, bin_edges = np.histogram(self.world_map.flatten(), bins=255, range=(min_z, max_z))
        
        water_pixel_threshold = int((self.percent_water / 100.0) * (self.x_range * self.y_range))
        
        count = 0
        threshold_bin = 0
        for i, num_pixels in enumerate(hist):
            count += num_pixels
            if count > water_pixel_threshold:
                threshold_bin = i
                break
        
        water_level_threshold = bin_edges[threshold_bin]

        # 5. Color Scaling
        # Create a new array for color indices
        color_map = np.zeros_like(self.world_map, dtype=np.uint8)

        water_mask = self.world_map < water_level_threshold
        land_mask = ~water_mask
        
        # Scale water colors
        water_min = min_z
        water_max = water_level_threshold
        if water_max == water_min: water_max = water_min + 1
        color_map[water_mask] = 1 + np.floor(14.99 * (self.world_map[water_mask] - water_min) / (water_max - water_min))

        # Scale land colors
        land_min = water_level_threshold
        land_max = max_z
        if land_max == land_min: land_max = land_min + 1
        color_map[land_mask] = 16 + np.floor(15.99 * (self.world_map[land_mask] - land_min) / (land_max - land_min))

        color_map = np.clip(color_map, 1, 31)
        self.world_map = color_map

        # 6. Apply Ice Caps
        if self.percent_ice > 0:
            total_filled = 0
            # North Pole
            for x in range(self.x_range):
                old_color = self.world_map[x, 0]
                if old_color < 32:
                    total_filled += self._flood_fill(x, 0, old_color, self.percent_ice / 2.0)
            
            # South Pole
            for x in range(self.x_range):
                old_color = self.world_map[x, self.y_range - 1]
                if old_color < 32:
                    total_filled += self._flood_fill(x, self.y_range - 1, old_color, self.percent_ice / 2.0)

        # 7. Create final image from color indices and palette
        rgb_map = np.zeros((self.y_range, self.x_range, 3), dtype=np.uint8)
        for i, color in enumerate(self.palette):
            # The map is (width, height), but images are addressed (height, width)
            # So we find where world_map transposed equals i.
            mask = self.world_map.T == i
            rgb_map[mask] = color
            
        return Image.fromarray(rgb_map, 'RGB')


class App(tk.Tk):
    """
    The main Tkinter application class.
    """
    def __init__(self):
        super().__init__()
        self.title("Fractal World Generator")
        self.geometry("1000x800")
        
        self.pil_image = None
        self.tk_image = None
        
        # --- Main Layout ---
        self.main_frame = ttk.Frame(self, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.controls_frame = ttk.Labelframe(self.main_frame, text="Controls", padding="10")
        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.image_frame = ttk.Labelframe(self.main_frame, text="Generated Map", padding="10")
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # --- Image Canvas ---
        self.canvas = tk.Canvas(self.image_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # --- Control Widgets ---
        self.params = {
            'width': tk.IntVar(value=640),
            'height': tk.IntVar(value=320),
            'seed': tk.IntVar(value=random.randint(0, 100000)),
            'faults': tk.IntVar(value=200),
            'water': tk.DoubleVar(value=60.0),
            'ice': tk.DoubleVar(value=10.0)
        }
        
        self._create_control_widgets()
        
    def _create_control_widgets(self):
        """Creates and lays out all the control widgets."""
        row = 0
        
        # Width and Height
        self._create_entry_widget(self.controls_frame, "Width:", self.params['width'], row)
        row += 1
        self._create_entry_widget(self.controls_frame, "Height:", self.params['height'], row)
        row += 1
        
        # Separator
        ttk.Separator(self.controls_frame, orient='horizontal').grid(row=row, columnspan=3, sticky='ew', pady=10)
        row += 1

        # Seed
        self._create_entry_widget(self.controls_frame, "Seed:", self.params['seed'], row, include_random_button=True)
        row += 1

        # Faults
        self._create_slider_widget(self.controls_frame, "Faults:", self.params['faults'], 1, 1000, row)
        row += 2

        # Water Percentage
        self._create_slider_widget(self.controls_frame, "Water %:", self.params['water'], 0, 100, row)
        row += 2
        
        # Ice Percentage
        self._create_slider_widget(self.controls_frame, "Ice %:", self.params['ice'], 0, 100, row)
        row += 2
        
        # Separator
        ttk.Separator(self.controls_frame, orient='horizontal').grid(row=row, columnspan=3, sticky='ew', pady=10)
        row += 1
        
        # Progress Bar
        self.progress = ttk.Progressbar(self.controls_frame, orient='horizontal', mode='determinate')
        self.progress.grid(row=row, columnspan=3, sticky='ew', pady=5)
        row += 1

        # Action Buttons
        button_frame = ttk.Frame(self.controls_frame)
        button_frame.grid(row=row, columnspan=3, pady=10)
        
        self.generate_button = ttk.Button(button_frame, text="Generate", command=self.start_generation)
        self.generate_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(button_frame, text="Save As...", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)

    def _create_entry_widget(self, parent, label_text, var, row, include_random_button=False):
        """Helper to create a label and an entry field."""
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        entry = ttk.Entry(parent, textvariable=var, width=10)
        entry.grid(row=row, column=1, sticky='ew', padx=5)
        if include_random_button:
            button = ttk.Button(parent, text="🎲", width=3, command=lambda: var.set(random.randint(0, 100000)))
            button.grid(row=row, column=2, sticky='w')
        return entry
        
    def _create_slider_widget(self, parent, label_text, var, from_, to, row):
        """Helper to create a label, slider, and entry field group."""
        ttk.Label(parent, text=label_text).grid(row=row, column=0, columnspan=3, sticky='w', padx=5)
        slider = ttk.Scale(parent, from_=from_, to=to, orient='horizontal', variable=var,
                           command=lambda val, v=var: v.set(float(val)))
        slider.grid(row=row+1, column=0, columnspan=2, sticky='ew', padx=5)
        entry = ttk.Entry(parent, textvariable=var, width=7)
        entry.grid(row=row+1, column=2, sticky='w', padx=5)
        
    def start_generation(self):
        """Handles the 'Generate' button click by starting a new thread."""
        self.generate_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        
        params_dict = {key: var.get() for key, var in self.params.items()}
        self.progress.config(maximum=params_dict['faults'])
        self.progress['value'] = 0
        
        # Run generation in a separate thread to not freeze the GUI
        thread = threading.Thread(target=self.run_generation_in_thread, args=(params_dict,), daemon=True)
        thread.start()
        
    def run_generation_in_thread(self, params_dict):
        """The function that runs in the background thread."""
        generator = FractalWorldGenerator(params_dict, progress_callback=self.update_progress)
        self.pil_image = generator.generate()
        self.after(0, self.update_canvas)

    def update_progress(self, value):
        """Updates the progress bar from the worker thread."""
        self.after(0, self.progress.config, {'value': value})

    def update_canvas(self):
        """Updates the canvas with the newly generated image."""
        if self.pil_image:
            # Resize image to fit the canvas while maintaining aspect ratio
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            img_w, img_h = self.pil_image.size
            
            # Prevent division by zero if canvas is not yet drawn
            if canvas_width == 1 or canvas_height == 1:
                self.after(100, self.update_canvas) # Try again later
                return

            scale = min(canvas_width / img_w, canvas_height / img_h)
            new_size = (int(img_w * scale), int(img_h * scale))

            resized_image = self.pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Must keep a reference to this photoimage object
            self.tk_image = ImageTk.PhotoImage(resized_image)
            
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width // 2, canvas_height // 2,
                                     anchor=tk.CENTER, image=self.tk_image)
        
        self.generate_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)

    def save_image(self):
        """Handles the 'Save As...' button click."""
        if not self.pil_image:
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("GIF Image", "*.gif"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                self.pil_image.save(file_path)
            except Exception as e:
                tk.messagebox.showerror("Save Error", f"Failed to save image:\n{e}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
