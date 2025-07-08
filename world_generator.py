# beanz-y/fractal_world_generator/fractal_world_generator-8b752999818ebdee7e3c696935b618f2a364ff8f/world_generator.py
import numpy as np # type: ignore
import random
import math
from PIL import Image # type: ignore
from utils import SimplexNoise
from constants import BIOME_DEFINITIONS, PREDEFINED_PALETTES, THEME_NAME_FRAGMENTS
import numba # type: ignore
from collections import deque
from scipy.ndimage import label, find_objects, distance_transform_edt # type: ignore


# --- Numba-Optimized Simplex Noise Functions ---
# This section contains a JIT-compiled implementation of the Simplex Noise logic.
# By moving these calculations into functions that Numba can optimize, we can
# significantly speed up the generation of moisture and ice noise maps.

F3 = 1.0 / 3.0
G3 = 1.0 / 6.0

@numba.jit(nopython=True, fastmath=True)
def _dot3_jit(grad, x, y, z):
    """Numba-compatible dot product for 3D vectors."""
    return grad[0] * x + grad[1] * y + grad[2] * z

@numba.jit(nopython=True, fastmath=True)
def _noise3_jit(x, y, z, perm, grad3):
    """Numba-compatible 3D Simplex Noise calculation."""
    s = (x + y + z) * F3
    i, j, k = int(math.floor(x + s)), int(math.floor(y + s)), int(math.floor(z + s))
    t = (i + j + k) * G3
    x0, y0, z0 = x - (i - t), y - (j - t), z - (k - t)

    if x0 >= y0:
        if y0 >= z0: i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 1, 0
        elif x0 >= z0: i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 0, 1
        else: i1, j1, k1, i2, j2, k2 = 0, 0, 1, 1, 0, 1
    else:
        if y0 < z0: i1, j1, k1, i2, j2, k2 = 0, 0, 1, 0, 1, 1
        elif x0 < z0: i1, j1, k1, i2, j2, k2 = 0, 1, 0, 0, 1, 1
        else: i1, j1, k1, i2, j2, k2 = 0, 1, 0, 1, 1, 0

    x1, y1, z1 = x0 - i1 + G3, y0 - j1 + G3, z0 - k1 + G3
    x2, y2, z2 = x0 - i2 + 2 * G3, y0 - j2 + 2 * G3, z0 - k2 + 2 * G3
    x3, y3, z3 = x0 - 1.0 + 3 * G3, y0 - 1.0 + 3 * G3, z0 - 1.0 + 3 * G3

    ii, jj, kk = i & 255, j & 255, k & 255
    
    gi0 = perm[ii + perm[jj + perm[kk]]] % 12
    gi1 = perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12
    gi2 = perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12
    gi3 = perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]] % 12

    t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0
    n0 = t0**4 * _dot3_jit(grad3[gi0], x0, y0, z0) if t0 > 0 else 0.0
    t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1
    n1 = t1**4 * _dot3_jit(grad3[gi1], x1, y1, z1) if t1 > 0 else 0.0
    t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2
    n2 = t2**4 * _dot3_jit(grad3[gi2], x2, y2, z2) if t2 > 0 else 0.0
    t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3
    n3 = t3**4 * _dot3_jit(grad3[gi3], x3, y3, z3) if t3 > 0 else 0.0
    
    return 32.0 * (n0 + n1 + n2 + n3)

@numba.jit(nopython=True, fastmath=True)
def _fractal_noise3_jit(x, y, z, perm, grad3, octaves, persistence, lacunarity):
    """Numba-compatible fractal noise (fBm)."""
    total, frequency, amplitude, max_value = 0.0, 1.0, 1.0, 0.0
    for _ in range(octaves):
        total += _noise3_jit(x * frequency, y * frequency, z * frequency, perm, grad3) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return total / max_value if max_value > 0 else 0

@numba.jit(nopython=True)
def scalar_clip(value, min_val, max_val):
    if value < min_val: return min_val
    elif value > max_val: return max_val
    else: return value

@numba.jit(nopython=True, fastmath=True)
def _numba_simulate_rainfall(height_map, land_mask, wind_direction):
    x_range, y_range = height_map.shape
    moisture_map = np.zeros_like(height_map, dtype=np.float32)

    # Simplified single-pass rainfall simulation
    if wind_direction < 2:  # Horizontal winds (East-West)
        for i in range(y_range):
            air_moisture = 1.0
            # Determine the direction of iteration
            x_iterator = range(x_range) if wind_direction == 0 else range(x_range - 1, -1, -1)
            for j in x_iterator:
                # Calculate previous pixel's x-coordinate with wrapping
                px = (j - 1 + x_range) % x_range if wind_direction == 0 else (j + 1) % x_range
                
                if not land_mask[j, i]:
                    air_moisture = min(1.0, air_moisture + 0.05)
                    continue

                air_moisture += 0.002
                elevation_diff = height_map[j, i] - height_map[px, i]

                precipitation = 0.015 * air_moisture
                if elevation_diff > 0:
                    precipitation += elevation_diff * 12.0 * air_moisture * 0.001
                
                precipitation = min(air_moisture, precipitation)
                moisture_map[j, i] = precipitation
                air_moisture = max(0, air_moisture - precipitation)
    else:  # Vertical winds (North-South)
        for i in range(x_range):
            air_moisture = 1.0
            y_iterator = range(y_range) if wind_direction == 2 else range(y_range - 1, -1, -1)
            for j in y_iterator:
                if (wind_direction == 2 and j == 0) or (wind_direction == 3 and j == y_range - 1):
                    py = j
                else:
                    py = j - 1 if wind_direction == 2 else j + 1

                if not land_mask[i, j]:
                    air_moisture = min(1.0, air_moisture + 0.05)
                    continue
                
                air_moisture += 0.002
                elevation_diff = height_map[i, j] - height_map[i, py]

                precipitation = 0.015 * air_moisture
                if elevation_diff > 0:
                    precipitation += elevation_diff * 12.0 * air_moisture * 0.001
                
                precipitation = min(air_moisture, precipitation)
                moisture_map[i, j] = precipitation
                air_moisture = max(0, air_moisture - precipitation)

    return moisture_map

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
        settlement_seed = self.params.get('seed') + 1 # Use a derived seed for settlements
        self.ice_noise = SimplexNoise(seed=ice_seed)
        self.moisture_noise = SimplexNoise(seed=moisture_seed)
        self.settlement_noise = SimplexNoise(seed=settlement_seed)
        
        self.moisture_perm = np.array(self.moisture_noise.perm, dtype=np.int64)
        self.moisture_grad3 = np.array(self.moisture_noise.grad3, dtype=np.float64)
        self.ice_perm = np.array(self.ice_noise.perm, dtype=np.int64)
        self.ice_grad3 = np.array(self.ice_noise.grad3, dtype=np.float64)
        self.settlement_perm = np.array(self.settlement_noise.perm, dtype=np.int64)
        self.settlement_grad3 = np.array(self.settlement_noise.grad3, dtype=np.float64)

        self.palette = palette
        self.INT_MIN_PLACEHOLDER = -2**31
        self.world_map = None
        self.color_map = None
        self.pil_image = None
        self.color_map_before_ice = None
        self.land_mask = None
        
        # Civilization data
        self.settlements = []
        self.natural_features = {}
        self.used_names = set()

    def set_ice_seed(self, seed):
        self.ice_noise = SimplexNoise(seed=seed)
        self.ice_perm = np.array(self.ice_noise.perm, dtype=np.int64)
        self.ice_grad3 = np.array(self.ice_noise.grad3, dtype=np.float64)

    @staticmethod
    @numba.jit(nopython=True, parallel=True, fastmath=True)
    def _numba_generate_noise_map(x_range, y_range, perm, grad3, scale, octaves, persistence, lacunarity):
        noise_map = np.empty((x_range, y_range), dtype=np.float32)
        
        two_pi = 2 * np.pi
        inv_two_pi = 1 / two_pi
        y_scale_factor = 2 / y_range if y_range > 0 else 0

        for y in numba.prange(y_range):
            for x in range(x_range):
                angle = two_pi * (x / x_range)
                nx = inv_two_pi * scale * np.cos(angle)
                nz = inv_two_pi * scale * np.sin(angle)
                ny = y * y_scale_factor * scale
                noise_map[x, y] = _fractal_noise3_jit(nx, ny, nz, perm, grad3, octaves, persistence, lacunarity)
        return noise_map

    @staticmethod
    @numba.jit(nopython=True)
    def _numba_add_fault_line(world_map, x_range, y_range, y_range_div_pi, y_range_div_2, sin_iter_phi, flag1, beta, alpha):
        cos_val = np.cos(alpha) * np.cos(beta)
        clipped_cos = scalar_clip(cos_val, -1.0, 1.0)
        tan_b = np.tan(np.arccos(clipped_cos))
        
        xsi = int(x_range / 2.0 - (x_range / np.pi) * beta)
        delta = 1 if flag1 else -1

        for phi in range(x_range):
            sin_val = sin_iter_phi[xsi - phi + x_range]
            theta = int((y_range_div_pi * np.arctan(sin_val * tan_b)) + y_range_div_2)
            theta = max(0, min(y_range - 1, theta))
            
            if world_map[phi, theta] == -2147483648:
                world_map[phi, theta] = delta
            else:
                world_map[phi, theta] += delta
        return world_map

    def _add_fault(self):
        flag1 = random.randint(0, 1)
        alpha, beta = (random.random() - 0.5) * np.pi, (random.random() - 0.5) * np.pi
        
        self.world_map = self._numba_add_fault_line(
            self.world_map, self.x_range, self.y_range, 
            self.y_range_div_pi, self.y_range_div_2, 
            self.sin_iter_phi, flag1, beta, alpha
        )

    @staticmethod
    @numba.jit(nopython=True)
    def _numba_apply_erosion(heightmap, x_range, y_range, erosion_passes):
        for _ in range(erosion_passes):
            # The core erosion logic remains the same
            source_map = heightmap.copy()
            for y in range(y_range):
                for x in range(x_range):
                    current_height = source_map[x, y]
                    lowest_neighbor_height = current_height
                    lowest_nx, lowest_ny = x, y

                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0: continue
                            nx, ny = (x + dx + x_range) % x_range, y + dy
                            if 0 <= ny < y_range:
                                if source_map[nx, ny] < lowest_neighbor_height:
                                    lowest_neighbor_height = source_map[nx, ny]
                                    lowest_nx, lowest_ny = nx, ny
                    
                    if lowest_neighbor_height < current_height:
                        sediment_amount = (current_height - lowest_neighbor_height) * 0.1
                        heightmap[x, y] -= sediment_amount
                        heightmap[lowest_nx, lowest_ny] += sediment_amount
        return heightmap

    def _apply_erosion(self, erosion_passes):
        if erosion_passes <= 0: return
        
        if self.progress_callback: self.progress_callback(30, "Applying erosion...")
        
        heightmap = self.world_map.astype(np.float32)
        
        self.world_map = self._numba_apply_erosion(heightmap, self.x_range, self.y_range, erosion_passes).astype(np.int32)
                
    def _apply_biomes(self, land_mask, land_min, land_max, temp_offset=0.0):
        if self.progress_callback: self.progress_callback(50, "Applying biomes: Calculating moisture...")
        y_coords = np.arange(self.y_range)
        latitude_temp = 1.0 - (np.abs(y_coords - self.y_range / 2.0) / (self.y_range / 2.0))
        
        altitude_norm = (self.world_map - land_min) / (land_max - land_min)
        altitude_temp_effect = self.params.get('altitude_temp_effect', 0.5)
        
        temperature = np.tile(latitude_temp, (self.x_range, 1)) - (altitude_norm * altitude_temp_effect)
        
        # Apply the global temperature offset for simulations
        temperature -= temp_offset
        temperature = np.clip(temperature, 0, 1)

        base_moisture = self._numba_generate_noise_map(
            self.x_range, self.y_range, self.moisture_perm, self.moisture_grad3,
            scale=5.0, octaves=4, persistence=0.5, lacunarity=2.0
        )
        base_moisture = (base_moisture - np.min(base_moisture)) / (np.max(base_moisture) - np.min(base_moisture))
        
        wind_direction_str = self.params.get('wind_direction', 'West to East')
        wind_map = {'West to East': 0, 'East to West': 1, 'North to South': 2, 'South to North': 3}
        wind_direction_code = wind_map.get(wind_direction_str, 0)
        simulated_moisture = _numba_simulate_rainfall(self.world_map, land_mask, wind_direction_code)
        
        max_sim_moisture = np.max(simulated_moisture)
        if max_sim_moisture > 0:
            simulated_moisture /= max_sim_moisture
            
        moisture = (base_moisture * 0.4) + (simulated_moisture * 0.6)
        
        max_total_moisture = np.max(moisture)
        if max_total_moisture > 0:
            moisture /= max_total_moisture

        if self.progress_callback: self.progress_callback(70, "Applying biomes: Classifying biomes...")

        lat_factor = np.abs(y_coords - self.y_range / 2.0) / (self.y_range / 2.0)
        non_polar_mask = (lat_factor < 0.85)
        
        alpine_props = BIOME_DEFINITIONS['alpine_glacier']
        alpine_mask = (altitude_norm > 0.7) & \
                      (temperature < alpine_props['temp_range'][1]) & \
                      land_mask & \
                      np.tile(non_polar_mask, (self.x_range, 1))

        if np.any(alpine_mask):
            base_color = alpine_props['idx']
            num_shades = alpine_props.get('shades', 1)
            alpine_altitudes = altitude_norm[alpine_mask]
            color_indices = base_color + np.floor(alpine_altitudes * (num_shades - 0.01))
            self.color_map[alpine_mask] = color_indices.astype(np.uint8)

        remaining_land_mask = land_mask & ~alpine_mask
        
        for name, props in BIOME_DEFINITIONS.items():
            if name == 'alpine_glacier' or name == 'glacier':
                continue

            t_min, t_max = props['temp_range']
            m_min, m_max = props['moist_range']
            
            biome_mask = (temperature >= t_min) & (temperature < t_max) & \
                         (moisture >= m_min) & (moisture < m_max) & \
                         remaining_land_mask

            if np.any(biome_mask):
                base_color = props['idx']
                num_shades = props.get('shades', 1)
                biome_altitudes = altitude_norm[biome_mask]
                color_indices = base_color + np.floor(biome_altitudes * (num_shades - 0.01))
                self.color_map[biome_mask] = color_indices.astype(np.uint8)
        
        uncolored_land_mask = (self.color_map == 0) & land_mask
        if np.any(uncolored_land_mask):
            tundra_props = BIOME_DEFINITIONS['tundra']
            base_color = tundra_props['idx']
            num_shades = tundra_props.get('shades', 1)
            uncolored_altitudes = altitude_norm[uncolored_land_mask]
            color_indices = base_color + np.floor(uncolored_altitudes * (num_shades - 0.01))
            self.color_map[uncolored_land_mask] = color_indices.astype(np.uint8)

    def _apply_ice_caps(self, percent_ice):
        if percent_ice <= 0: return
        if self.progress_callback: self.progress_callback(80, "Applying ice caps: Calculating noise...")

        noise_map = self._numba_generate_noise_map(
            self.x_range, self.y_range, self.ice_perm, self.ice_grad3,
            scale=4.0, octaves=6, persistence=0.5, lacunarity=2.0
        )
        noise_map = (noise_map - np.min(noise_map)) / (np.max(noise_map) - np.min(noise_map))

        if self.progress_callback: self.progress_callback(90, "Applying ice caps: Forming ice...")
        
        y_coords = np.arange(self.y_range)
        latitude_factor_1d = np.abs(y_coords - (self.y_range - 1) / 2.0) / (self.y_range / 2.0)
        latitude_factor = np.tile(latitude_factor_1d, (self.x_range, 1))

        ice_score_map = (1.2 * latitude_factor) + (0.4 * noise_map)
        
        valid_pixels_mask = (self.color_map != 0)

        if not np.any(valid_pixels_mask): return

        ice_threshold = np.percentile(ice_score_map[valid_pixels_mask], 100 - percent_ice)
        ice_mask = (ice_score_map >= ice_threshold) & valid_pixels_mask
        if not np.any(ice_mask): return

        ice_altitudes = self.world_map[ice_mask]
        min_ice_alt, max_ice_alt = np.min(ice_altitudes), np.max(ice_altitudes)
        if max_ice_alt == min_ice_alt: max_ice_alt = min_ice_alt + 1
        
        ice_base_color_start = BIOME_DEFINITIONS['glacier']['idx']
        ice_num_base_colors = BIOME_DEFINITIONS['glacier']['shades']
        
        original_colors_under_ice = self.color_map[ice_mask]
        
        normalized_ice_alts = (ice_altitudes - min_ice_alt) / (max_ice_alt - min_ice_alt)
        base_ice_colors = ice_base_color_start + np.floor(normalized_ice_alts * (ice_num_base_colors - 0.01))
        
        terrain_modifier = (original_colors_under_ice % 2)
        final_ice_colors = base_ice_colors - terrain_modifier
        
        self.color_map[ice_mask] = final_ice_colors.astype(np.uint8)
    
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
        
        # --- POI Generation ---
        self.used_names = set() # Reset used names for each generation
        self.natural_features = self.generate_natural_features()
        self.generate_settlements()
        
        if self.progress_callback: self.progress_callback(99, "Creating image...")
        self.pil_image = self.create_image()
        
        if self.progress_callback: self.progress_callback(100, "Done.")
        return self.pil_image

    def finalize_map(self, percent_water, percent_ice, map_style, temp_offset=0.0):
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
        self.land_mask = ~water_mask
        
        water_min, water_max = min_z, water_level_threshold
        if water_max == water_min: water_max = water_min + 1
        self.color_map[water_mask] = 1 + np.floor(6.99 * (self.world_map[water_mask] - water_min) / (water_max - water_min))

        land_min, land_max = water_level_threshold, max_z
        if land_max == land_min: land_max = land_min + 1
        
        if map_style == 'Biome':
            self._apply_biomes(self.land_mask, land_min, land_max, temp_offset)
        else: # Terrain style
            normalized_altitude = (self.world_map[self.land_mask] - land_min) / (land_max - land_min)
            self.color_map[self.land_mask] = 16 + np.floor(15.99 * normalized_altitude)

        self.color_map_before_ice = self.color_map.copy()
        self._apply_ice_caps(percent_ice)

    def _calculate_distance_to_water(self, land_mask):
        """
        Calculates the distance from each land pixel to the nearest water pixel
        using scipy's fast Euclidean Distance Transform.
        """
        if self.progress_callback: self.progress_callback(96, "Placing settlements: Calculating water distance...")
        
        # This one line replaces the entire BFS queue and loop.
        # It calculates the exact Euclidean distance from each land pixel (True in the mask)
        # to the nearest water pixel (False in the mask).
        return distance_transform_edt(land_mask)

    def _generate_fantasy_name(self, name_fragments, max_retries=10):
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
            
            if name not in self.used_names:
                self.used_names.add(name)
                return name.capitalize()

        # Failsafe if a unique name isn't found
        return f"{random.choice(name_fragments['prefixes'])}{random.choice(name_fragments['suffixes'])}".capitalize()

    def generate_settlements(self):
        if self.world_map is None or self.land_mask is None: return
        num_settlements = self.params['num_settlements']
        if num_settlements == 0:
            self.settlements = []
            return

        if self.progress_callback: self.progress_callback(98, "Placing settlements: Calculating suitability...")

        BIOME_SUITABILITY = {
            'temperate_forest': 1.0, 'temperate_rainforest': 0.9,
            'shrubland': 0.8, 'savanna': 0.7,
            'tropical_forest': 0.6, 'tropical_rainforest': 0.5,
            'taiga': 0.4, 'desert': 0.2, 'tundra': 0.1
        }

        # Start with a base suitability map of all zeros.
        suitability = np.zeros(self.world_map.shape, dtype=np.float32)
        
        # 1. Establish BASE suitability from biomes. Ice/Glaciers will have a score of 0.
        for name, props in BIOME_DEFINITIONS.items():
            if name in BIOME_SUITABILITY:
                start_idx, end_idx = props['idx'], props['idx'] + props.get('shades', 1)
                biome_mask = (self.color_map >= start_idx) & (self.color_map < end_idx)
                suitability[biome_mask] = BIOME_SUITABILITY[name]
        
        # --- MODIFICATION START: Corrected Application of Modifiers ---

        # 2. Apply global modifiers using broadcasting. This is simpler and avoids indexing errors.
        
        # Apply Latitude Modifier to the entire map.
        # Uninhabitable areas (suitability=0) will remain 0.
        y_coords = np.arange(self.y_range)
        latitude_score = 1.0 - (np.abs(y_coords - self.y_range / 2.0) / (self.y_range / 2.0))
        suitability *= latitude_score[np.newaxis, :]

        # 3. Create a final mask of habitable land to apply subsequent, more complex modifiers.
        habitable_mask = suitability > 0

        # Apply Water Proximity Modifier only to habitable areas
        if np.any(habitable_mask):
            distance_to_water = self._calculate_distance_to_water(self.land_mask)
            water_bonus_mask = habitable_mask & np.isfinite(distance_to_water)
            suitability[water_bonus_mask] += 0.8 / ((distance_to_water[water_bonus_mask]**0.5) + 1e-6)

        # Apply Altitude and Slope Modifiers only to habitable areas
        # Re-filter the habitable mask in case water bonus made some areas negative
        habitable_mask = suitability > 0
        
        land_altitudes = self.world_map[habitable_mask]
        if land_altitudes.size > 0:
            min_land_alt, max_land_alt = np.min(land_altitudes), np.max(land_altitudes)
            if max_land_alt > min_land_alt:
                normalized_land_alt = (self.world_map[habitable_mask] - min_land_alt) / (max_land_alt - min_land_alt)
                suitability[habitable_mask] -= normalized_land_alt * 0.7

        grad_x, grad_y = np.gradient(self.world_map.astype(np.float32))
        slope = np.sqrt(grad_x**2 + grad_y**2)
        if np.max(slope) > 0:
            normalized_slope = slope / np.max(slope)
            # Make sure we're still only applying to habitable areas
            habitable_mask = suitability > 0
            suitability[habitable_mask] -= normalized_slope[habitable_mask] * 1.5

        # Apply Noise Modifier
        habitable_mask = suitability > 0
        settlement_noise_map = self._numba_generate_noise_map(
            self.x_range, self.y_range, self.settlement_perm, self.settlement_grad3,
            scale=20.0, octaves=4, persistence=0.5, lacunarity=2.0
        )
        suitability[habitable_mask] += ((settlement_noise_map[habitable_mask] + 1.0) / 2.0) * 0.2
        
        # --- MODIFICATION END ---

        suitability[suitability < 0] = 0

        # Find the best settlement locations from the final suitability map.
        flat_suitability = suitability.flatten()
        
        possible_indices = np.where(flat_suitability > 0)[0]
        if possible_indices.size == 0:
            self.settlements = []
            return

        num_candidates = min(possible_indices.size, num_settlements * 20)
        
        candidate_values = flat_suitability[possible_indices]
        best_candidate_indices_in_subset = np.argpartition(candidate_values, -num_candidates)[-num_candidates:]
        original_indices = possible_indices[best_candidate_indices_in_subset]

        min_dist = math.sqrt(self.x_range * self.y_range) / (num_settlements * 0.5 + 10)
        min_dist_sq = min_dist**2

        self.settlements = []
        sorted_candidates = original_indices[np.argsort(flat_suitability[original_indices])][::-1]
        
        theme = self.params.get('theme', 'High Fantasy')
        name_fragments = THEME_NAME_FRAGMENTS.get(theme, THEME_NAME_FRAGMENTS['High Fantasy'])
        
        for idx in sorted_candidates:
            if len(self.settlements) >= num_settlements: break
            x, y = np.unravel_index(idx, (self.x_range, self.y_range))
            
            is_far_enough = True
            for ex_settlement in self.settlements:
                ex, ey = ex_settlement['x'], ex_settlement['y']
                dx = abs(x - ex); dx = min(dx, self.x_range - dx)
                dist_sq = dx**2 + (y - ey)**2
                if dist_sq < min_dist_sq:
                    is_far_enough = False; break
            
            if is_far_enough:
                name = self._generate_fantasy_name(name_fragments)
                stype = random.choice(name_fragments['types'])
                self.settlements.append({'x': x, 'y': y, 'name': name, 'type': 'settlement', 'stype': stype})

    def _find_features(self, mask, min_size=50):
        """
        Finds contiguous features in a mask using scipy.ndimage.label for high performance.
        This version has corrected axis handling.
        """
        features = []
        # The mask has shape (width, height), which is (x, y).
        labeled_array, num_features = label(mask)

        if num_features > 0:
            # Slices are returned as (slice_for_axis_0, slice_for_axis_1), so (slice_x, slice_y)
            slices = find_objects(labeled_array)
            for i, (slice_x, slice_y) in enumerate(slices):
                feature_label = i + 1
                feature_mask = labeled_array[slice_x, slice_y] == feature_label
                size = np.sum(feature_mask)

                if size >= min_size:
                    # np.argwhere returns coordinates as (row, col) relative to the slice.
                    # For our (x, y) mask slice, this is (local_x, local_y).
                    coords = np.argwhere(feature_mask)
                    # Calculate the mean for each axis correctly.
                    mean_x, mean_y = np.mean(coords, axis=0)
                    
                    # Convert the relative center to absolute map coordinates.
                    abs_center_x = int(mean_x + slice_x.start)
                    abs_center_y = int(mean_y + slice_y.start)

                    features.append({
                        'size': int(size),
                        'center': (abs_center_x, abs_center_y)
                    })

        features.sort(key=lambda f: f['size'], reverse=True)
        return features

    def generate_natural_features(self):
        if self.world_map is None: return {}
        if self.progress_callback: self.progress_callback(97, "Naming natural features...")

        num_features_to_find = self.params.get('num_features', 10)
        theme = self.params.get('theme', 'High Fantasy')
        name_fragments = THEME_NAME_FRAGMENTS.get(theme, THEME_NAME_FRAGMENTS['High Fantasy'])
        
        features = {'peaks': [], 'ranges': [], 'areas': []}
        
        # --- MODIFICATION START: Dynamic Minimum Feature Size ---
        total_land_pixels = np.sum(self.land_mask)
        if total_land_pixels == 0:
            # If there's no land, we can't name land features.
            # We can still name the ocean if that's all there is.
            water_bodies = self._find_features(~self.land_mask, min_size=1)
            if water_bodies:
                ocean_name = f"The {self._generate_fantasy_name(name_fragments)} Ocean"
                water_bodies[0]['name'] = ocean_name
                features['areas'].append(water_bodies[0])
            return features
        
        # Calculate min_area_size based on a percentage of the LAND area,
        # ensuring it doesn't get too small on tiny islands.
        min_land_feature_size = max(50, int(total_land_pixels * 0.01))
        # --- MODIFICATION END ---


        # --- Step 1: Find all potential features of all types ---
        
        all_candidates = []

        # Find water bodies
        # Water bodies are still compared to the total map size, as they can be vast.
        min_water_size = int(self.x_range * self.y_range * 0.002)
        water_bodies = self._find_features(~self.land_mask, min_size=1)
        if water_bodies:
            all_candidates.append({'feature': water_bodies.pop(0), 'type': 'ocean'})
            for body in water_bodies:
                if body['size'] < min_water_size: continue
                center_x, center_y = body['center']
                if (self.land_mask[np.clip(center_x - 5, 0, self.x_range-1), center_y] and
                    self.land_mask[np.clip(center_x + 5, 0, self.x_range-1), center_y] and
                    self.land_mask[center_x, np.clip(center_y - 5, 0, self.y_range-1)] and
                    self.land_mask[center_x, np.clip(center_y + 5, 0, self.y_range-1)]):
                    all_candidates.append({'feature': body, 'type': 'lake'})
                else:
                    all_candidates.append({'feature': body, 'type': 'sea'})
        
        # Find mountain ranges using the new dynamic land feature size
        min_mountain_height = np.percentile(self.world_map[self.land_mask], 90)
        mountain_mask = (self.world_map >= min_mountain_height) & self.land_mask
        for feature in self._find_features(mountain_mask, min_size=min_land_feature_size):
            all_candidates.append({'feature': feature, 'type': 'range'})

        # Find major biome areas using the new dynamic land feature size
        biome_types_to_find = { 'desert': 'desert', 'tropical_rainforest': 'jungle', 
                                'temperate_forest': 'forest', 'tundra': 'wastes'}
        for biome_key, f_type in biome_types_to_find.items():
            props = BIOME_DEFINITIONS[biome_key]
            mask = (self.color_map_before_ice >= props['idx']) & (self.color_map_before_ice < props['idx'] + props.get('shades', 1))
            for feature in self._find_features(mask, min_size=min_land_feature_size):
                 all_candidates.append({'feature': feature, 'type': f_type})

        # --- Step 2: Sort all candidates by size and name them ---

        all_candidates.sort(key=lambda x: x['feature']['size'], reverse=True)

        named_features_count = 0
        named_types = set()
        named_centers = []
        min_dist_sq = (self.x_range / (num_features_to_find + 10))**2

        for cand in all_candidates:
            if named_features_count >= num_features_to_find: break
            
            center = tuple(cand['feature']['center'])
            f_type = cand['type']
            
            is_new_type = f_type not in named_types
            is_far_enough = not named_centers or min(np.sum((np.array(center) - np.array(c))**2) for c in named_centers) > min_dist_sq

            if is_new_type or is_far_enough:
                name = ""
                if f_type == 'ocean': name = f"The {self._generate_fantasy_name(name_fragments)} Ocean"
                elif f_type == 'sea': name = f"The {self._generate_fantasy_name(name_fragments)} Sea"
                elif f_type == 'lake': name = f"Lake {self._generate_fantasy_name(name_fragments)}"
                elif f_type == 'range': name = f"The {self._generate_fantasy_name(name_fragments)} Mountains"
                elif f_type == 'desert': name = f"The {self._generate_fantasy_name(name_fragments)} Desert"
                elif f_type == 'jungle': name = f"The {self._generate_fantasy_name(name_fragments)} Jungle"
                elif f_type == 'forest': name = f"The {self._generate_fantasy_name(name_fragments)} Forest"
                elif f_type == 'wastes': name = f"The {self._generate_fantasy_name(name_fragments)} Wastes"

                final_feature = cand['feature']
                final_feature['name'] = name
                final_feature['type'] = f_type

                if f_type == 'range':
                    features['ranges'].append(final_feature)
                    labeled_mountains, _ = label(mountain_mask)
                    center_x, center_y = final_feature['center']
                    label_at_center = labeled_mountains[center_x, center_y]
                    if label_at_center > 0:
                        range_coords = np.argwhere(labeled_mountains == label_at_center)
                        range_heights = self.world_map[range_coords[:, 0], range_coords[:, 1]]
                        peak_x, peak_y = range_coords[np.argmax(range_heights)]
                        peak_name = f"Mount {self._generate_fantasy_name(name_fragments)}"
                        features['peaks'].append({'x': peak_x, 'y': peak_y, 'name': peak_name, 'type': 'peak'})
                else:
                    features['areas'].append(final_feature)
                
                named_features_count += 1
                named_types.add(f_type)
                named_centers.append(center)
                
        return features

    def create_image(self):
        if self.color_map is None: return None
        clipped_color_map = np.clip(self.color_map, 0, len(self.palette) - 1)
        rgb_map = np.zeros((self.y_range, self.x_range, 3), dtype=np.uint8)
        for i, color in enumerate(self.palette):
            mask = clipped_color_map.T == i
            rgb_map[mask] = color
        return Image.fromarray(rgb_map, 'RGB')