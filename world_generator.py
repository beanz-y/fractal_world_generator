import numpy as np # type: ignore
import random
import math
from PIL import Image # type: ignore
from utils import SimplexNoise

# New Biome Definitions based on Whittaker model
# Ranges are for (temperature, moisture) from 0.0 to 1.0
# 'idx' is the base color index, 'shades' is the number of colors for altitude shading
BIOME_DEFINITIONS = {
    'glacier':              {'idx': 52, 'shades': 4, 'temp_range': (0.0, 0.125), 'moist_range': (0.0, 1.0)},
    'tundra':               {'idx': 48, 'shades': 4, 'temp_range': (0.125, 0.25), 'moist_range': (0.0, 1.0)},
    'taiga':                {'idx': 40, 'shades': 8, 'temp_range': (0.25, 0.5), 'moist_range': (0.0, 0.33)},
    'shrubland':            {'idx': 24, 'shades': 4, 'temp_range': (0.25, 0.5), 'moist_range': (0.33, 0.66)},
    'temperate_forest':     {'idx': 32, 'shades': 8, 'temp_range': (0.25, 0.5), 'moist_range': (0.66, 1.0)},
    'desert':               {'idx': 16, 'shades': 8, 'temp_range': (0.5, 1.0), 'moist_range': (0.0, 0.16)},
    'savanna':              {'idx': 25, 'shades': 4, 'temp_range': (0.5, 1.0), 'moist_range': (0.16, 0.33)},
    'tropical_forest':      {'idx': 33, 'shades': 4, 'temp_range': (0.5, 0.75), 'moist_range': (0.33, 0.66)},
    'temperate_rainforest': {'idx': 34, 'shades': 4, 'temp_range': (0.5, 0.75), 'moist_range': (0.66, 1.0)},
    'tropical_rainforest':  {'idx': 35, 'shades': 4, 'temp_range': (0.75, 1.0), 'moist_range': (0.33, 1.0)},
}


PREDEFINED_PALETTES = {
    "Biome": [
        # Water (0-15)
        (0,0,0), (28,82,106), (43,105,128), (57,128,149), (72,150,171), (86,173,192), (101,196,214), (115,219,235),
        (0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),
        # Desert (16-23)
        (249,225,184), (244,215,165), (239,205,146), (234,195,127), (229,185,108), (224,175,89), (219,165,70), (214,155,51),
        # Savanna / Shrubland (24-31)
        (185,209,139), (170,198,121), (155,187,103), (140,176,85), (125,165,67), (110,154,49), (95,143,31), (80,132,13),
        # Forest (32-39)
        (134,188,128), (119,173,113), (104,158,98), (89,143,83), (74,128,68), (59,113,53), (44,98,38), (29,83,23),
        # Taiga (40-47)
        (180,191,170), (171,181,162), (162,171,153), (153,161,145), (144,151,136), (135,141,128), (126,131,119), (117,121,111),
        # Tundra / Rock (48-51)
        (136,136,136), (150,150,150), (164,164,164), (178,178,178),
        # Glacier / Polar Ice (52-55)
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
        
        if self.progress_callback: self.progress_callback(30, "Applying erosion...")
        heightmap = self.world_map.astype(np.float32)
        for i in range(erosion_passes):
            if self.progress_callback:
                progress = 30 + int(10 * ((i + 1) / erosion_passes))
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
        if self.progress_callback: self.progress_callback(50, "Applying biomes: Calculating moisture...")
        # Temperature is based on latitude (y-axis) and altitude
        y_coords = np.arange(self.y_range)
        latitude_temp = 1.0 - (np.abs(y_coords - self.y_range / 2.0) / (self.y_range / 2.0))
        latitude_temp = np.tile(latitude_temp, (self.x_range, 1))

        altitude_norm = (self.world_map - land_min) / (land_max - land_min)
        altitude_temp_effect = self.params.get('altitude_temp_effect', 0.5)
        
        temperature = latitude_temp - (altitude_norm * altitude_temp_effect)
        temperature = np.clip(temperature, 0, 1)

        # Moisture is based on simplex noise
        moisture = np.zeros_like(self.world_map, dtype=np.float32)
        scale = 3.0
        for y in range(self.y_range):
            if self.progress_callback and y % 20 == 0:
                self.progress_callback(50 + int(20 * (y / self.y_range)), "Applying biomes: Calculating moisture...")
            for x in range(self.x_range):
                angle = 2 * math.pi * (x / self.x_range)
                nx = (1 / (2*math.pi)) * scale * math.cos(angle)
                nz = (1 / (2*math.pi)) * scale * math.sin(angle)
                ny = y / (self.y_range/2) * scale
                moisture[x,y] = self.moisture_noise.fractal_noise3(nx, ny, nz, 5, 0.5, 2.0)
        moisture = (moisture - np.min(moisture)) / (np.max(moisture) - np.min(moisture))

        if self.progress_callback: self.progress_callback(70, "Applying biomes: Classifying biomes...")
        # Classify and color biomes
        for name, props in BIOME_DEFINITIONS.items():
            t_min, t_max = props['temp_range']
            m_min, m_max = props['moist_range']
            
            biome_mask = (temperature >= t_min) & (temperature < t_max) & \
                         (moisture >= m_min) & (moisture < m_max) & \
                         land_mask

            if np.any(biome_mask):
                base_color = props['idx']
                num_shades = props.get('shades', 1)
                
                biome_altitudes = altitude_norm[biome_mask]
                
                color_indices = base_color + np.floor(biome_altitudes * (num_shades - 0.01))
                
                self.color_map[biome_mask] = color_indices.astype(np.uint8)


    def _apply_ice_caps(self, percent_ice):
        if percent_ice <= 0: return
        if self.progress_callback: self.progress_callback(80, "Applying ice caps: Calculating noise...")

        noise_map = np.zeros((self.x_range, self.y_range))
        scale = 4.0
        for y in range(self.y_range):
            if self.progress_callback and y % 20 == 0:
                self.progress_callback(80 + int(10 * (y/self.y_range)), "Applying ice caps: Calculating noise...")
            for x in range(self.x_range):
                angle = 2 * math.pi * (x / self.x_range)
                nx = (1 / (2*math.pi)) * scale * math.cos(angle)
                nz = (1 / (2*math.pi)) * scale * math.sin(angle)
                ny = y / (self.y_range/2) * scale
                noise_map[x, y] = self.ice_noise.fractal_noise3(nx, ny, nz, 6, 0.5, 2.0)
        noise_map = (noise_map - np.min(noise_map)) / (np.max(noise_map) - np.min(noise_map))

        if self.progress_callback: self.progress_callback(90, "Applying ice caps: Forming ice...")
        
        y_coords = np.arange(self.y_range)
        latitude_factor_1d = np.abs(y_coords - (self.y_range - 1) / 2.0) / (self.y_range / 2.0)
        latitude_factor = np.tile(latitude_factor_1d, (self.x_range, 1))

        ice_score_map = (1.2 * latitude_factor) + (0.4 * noise_map)
        
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
            if self.progress_callback and i % 20 == 0:
                self.progress_callback(int(25 * (i / num_faults)), f"Generating faults {i}/{num_faults}...")
            self._add_fault()
        
        if self.progress_callback: self.progress_callback(25, "Calculating heightmap...")
        temp_map = self.world_map.copy()
        temp_map[temp_map == self.INT_MIN_PLACEHOLDER] = 0
        self.world_map = np.cumsum(temp_map, axis=1)
        
        self._apply_erosion(self.params.get('erosion'))
        
        if self.progress_callback: self.progress_callback(50, "Finalizing map...")
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