# beanz-y/fractal_world_generator/fractal_world_generator-28f75751b57dacf83432892d2293f1e3754a3ba6/world_generator.py
#
# --- CHANGELOG ---
# 1. Revert: Plate Tectonics model has been temporarily removed to focus on other features.
# 2. Feature: River Generation
#    - Implemented a `_generate_rivers` method that carves river networks into the landscape.
#    - Rivers originate in high-altitude, high-moisture areas and flow to the sea.
#    - Settlement generation is now heavily influenced by proximity to rivers.
# 3. Feature: Enhanced Climate Simulation
#    - Added `_simulate_ocean_currents` to create more realistic sea temperatures.
#    - Introduced a "Season" parameter to affect global temperatures and ice coverage.
#    - Improved the rainfall simulation for more pronounced rain shadows.
# 4. Bugfix: Corrected a ValueError in `_apply_biomes` by fixing array broadcasting for temperature calculation.
# 5. Bugfix: Fixed an IndexError in `generate_natural_features` by correcting the peak-finding logic to handle array axes properly.
# 6. Bugfix: Corrected coordinate calculation in `_find_features` to prevent misplacing feature names.
# 7. Enhancement: Realistic River and Lake Generation
#    - Replaced the simple river algorithm with a hydrologically-sound model.
#    - Implemented a depression-filling algorithm to create inland lakes where water would naturally pool.
#    - Rivers are now guaranteed to flow to a lake or the ocean, carving channels into the terrain as they go.
# 8. Enhancement: Controllable Lake Coverage & Diverse River Origins
#    - Added a `lake_coverage` parameter to control what percentage of potential lakes are formed.
#    - Diversified river origins by lowering the elevation requirement, allowing rivers to form in wet, non-mountainous regions.
# 9. Bugfix: Correctly classify inland lakes vs. oceans/seas in the feature naming process.
# 10. Enhancement: Prevented lakes from forming in desert biomes and added control over the number of named lakes.
# 11. Bugfix: Passed the `labeled_water` map to the main app to enable accurate tooltips for water bodies.
# -----------------

import numpy as np
import random
import math
from PIL import Image
from utils import SimplexNoise, generate_contextual_name, generate_fantasy_name
from constants import BIOME_DEFINITIONS, THEME_NAME_FRAGMENTS
import numba
from collections import deque
from scipy.ndimage import label, find_objects, distance_transform_edt
import heapq


# --- Numba-Optimized Helper Functions ---
@numba.jit(nopython=True, fastmath=True)
def _dot3_jit(grad, x, y, z):
    return grad[0] * x + grad[1] * y + grad[2] * z

@numba.jit(nopython=True, fastmath=True)
def _noise3_jit(x, y, z, perm, grad3):
    s = (x + y + z) * (1.0 / 3.0)
    i, j, k = int(math.floor(x + s)), int(math.floor(y + s)), int(math.floor(z + s))
    t = (i + j + k) * (1.0 / 6.0)
    x0, y0, z0 = x - (i - t), y - (j - t), z - (k - t)

    if x0 >= y0:
        if y0 >= z0: i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 1, 0
        elif x0 >= z0: i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 0, 1
        else: i1, j1, k1, i2, j2, k2 = 0, 0, 1, 1, 0, 1
    else:
        if y0 < z0: i1, j1, k1, i2, j2, k2 = 0, 0, 1, 0, 1, 1
        elif x0 < z0: i1, j1, k1, i2, j2, k2 = 0, 1, 0, 0, 1, 1
        else: i1, j1, k1, i2, j2, k2 = 0, 1, 0, 1, 1, 0

    x1, y1, z1 = x0 - i1 + (1.0 / 6.0), y0 - j1 + (1.0 / 6.0), z0 - k1 + (1.0 / 6.0)
    x2, y2, z2 = x0 - i2 + (2.0 / 6.0), y0 - j2 + (2.0 / 6.0), z0 - k2 + (2.0 / 6.0)
    x3, y3, z3 = x0 - 1.0 + (3.0 / 6.0), y0 - 1.0 + (3.0 / 6.0), z0 - 1.0 + (3.0 / 6.0)
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
def _numba_simulate_rainfall(height_map, land_mask, wind_direction, ocean_temp_modifier):
    x_range, y_range = height_map.shape
    moisture_map = np.zeros_like(height_map, dtype=np.float32)
    
    # Wind direction mapping: 0: W->E, 1: E->W, 2: N->S, 3: S->N
    if wind_direction < 2: # Horizontal winds
        for i in range(y_range):
            air_moisture = 1.0
            x_iterator = range(x_range) if wind_direction == 0 else range(x_range - 1, -1, -1)
            for j in x_iterator:
                if not land_mask[j, i]:
                    # Evaporation from ocean is affected by temperature
                    air_moisture = min(1.5, air_moisture + 0.05 * (1 + ocean_temp_modifier[j, i]))
                    continue
                
                air_moisture += 0.002 # Base moisture pickup over land
                
                # Orographic lift (rain from mountains)
                px = (j - 1 + x_range) % x_range if wind_direction == 0 else (j + 1) % x_range
                elevation_diff = height_map[j, i] - height_map[px, i]
                precipitation = 0.015 * air_moisture
                if elevation_diff > 0:
                    # Increased precipitation on windward side of mountains
                    precipitation += elevation_diff * 20.0 * air_moisture * 0.001
                
                precipitation = min(air_moisture, precipitation)
                moisture_map[j, i] = precipitation
                air_moisture = max(0, air_moisture - precipitation)
    else: # Vertical winds
        for i in range(x_range):
            air_moisture = 1.0
            y_iterator = range(y_range) if wind_direction == 2 else range(y_range - 1, -1, -1)
            for j in y_iterator:
                if not land_mask[i, j]:
                    air_moisture = min(1.5, air_moisture + 0.05 * (1 + ocean_temp_modifier[i, j]))
                    continue
                
                air_moisture += 0.002
                
                py = j - 1 if wind_direction == 2 else j + 1
                if not (0 <= py < y_range): py = j
                elevation_diff = height_map[i, j] - height_map[i, py]
                precipitation = 0.015 * air_moisture
                if elevation_diff > 0:
                    precipitation += elevation_diff * 20.0 * air_moisture * 0.001
                
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
        
        # Noise generators
        ice_seed = self.params.get('ice_seed', self.seed)
        moisture_seed = self.params.get('moisture_seed', self.seed)
        settlement_seed = self.params.get('seed') + 1
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
        self.ocean_temp_modifier = None
        self.labeled_water = None
        
        self.settlements = []
        self.natural_features = {}
        self.major_features = []
        self.used_names = set()
        self.rivers = []

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
                nx, nz = inv_two_pi * scale * np.cos(angle), inv_two_pi * scale * np.sin(angle)
                ny = y * y_scale_factor * scale
                noise_map[x, y] = _fractal_noise3_jit(nx, ny, nz, perm, grad3, octaves, persistence, lacunarity)
        return noise_map

    def _generate_fractal_height_map(self):
        if self.progress_callback: self.progress_callback(0, "Generating faults...")
        self.world_map = np.full((self.x_range, self.y_range), self.INT_MIN_PLACEHOLDER, dtype=np.int32)
        y_range_div_2, y_range_div_pi = self.y_range / 2.0, self.y_range / np.pi
        sin_iter_phi = np.sin(np.arange(2 * self.x_range) * 2 * np.pi / self.x_range)
        
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

        for i in range(self.params['faults']):
            if self.progress_callback and i % 20 == 0: self.progress_callback(int(25*(i/self.params['faults'])), f"Generating faults {i}/{self.params['faults']}...")
            flag1 = random.randint(0, 1)
            alpha, beta = (random.random() - 0.5) * np.pi, (random.random() - 0.5) * np.pi
            self.world_map = _numba_add_fault_line(self.world_map, self.x_range, self.y_range, y_range_div_pi, y_range_div_2, sin_iter_phi, flag1, beta, alpha)
        
        if self.progress_callback: self.progress_callback(25, "Calculating heightmap...")
        temp_map = self.world_map.copy(); temp_map[temp_map == self.INT_MIN_PLACEHOLDER] = 0
        self.world_map = np.cumsum(temp_map, axis=1)
        return self.world_map

    @staticmethod
    @numba.jit(nopython=True)
    def _numba_apply_erosion(heightmap, x_range, y_range, erosion_passes):
        for _ in range(erosion_passes):
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
                            if 0 <= ny < y_range and source_map[nx, ny] < lowest_neighbor_height:
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

    def _simulate_ocean_currents(self):
        if self.progress_callback: self.progress_callback(40, "Simulating ocean currents...")
        self.ocean_temp_modifier = np.zeros_like(self.world_map, dtype=np.float32)
        if not hasattr(self, 'land_mask'): return

        # Base temperature gradient: warmer at equator
        y_coords = np.arange(self.y_range)
        equatorial_heat = np.cos(np.linspace(-np.pi/2, np.pi/2, self.y_range))**2
        self.ocean_temp_modifier = np.tile(equatorial_heat, (self.x_range, 1))
        
        # Simple current simulation
        for _ in range(5): # Iterations to spread heat
            current_temps = self.ocean_temp_modifier.copy()
            for x in range(self.x_range):
                for y in range(self.y_range):
                    if self.land_mask[x, y]:
                        self.ocean_temp_modifier[x, y] = -1 # Mark land as cold
                        continue
                    
                    # Gyre-like flow (simplified)
                    flow_y = (self.y_range/2 - y) * 0.01 # Towards equator
                    flow_x = 1 if y < self.y_range/2 else -1 # Westward/Eastward flow
                    
                    # Average with neighbors
                    avg_temp = 0
                    num_neighbors = 0
                    for dx, dy in [(int(flow_x), int(flow_y)), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = (x + dx + self.x_range) % self.x_range, y + dy
                        if 0 <= ny < self.y_range and not self.land_mask[nx, ny]:
                            avg_temp += current_temps[nx, ny]
                            num_neighbors += 1
            
                    if num_neighbors > 0:
                        self.ocean_temp_modifier[x, y] = (current_temps[x, y] * 2 + avg_temp) / (2 + num_neighbors)

        # Normalize modifier for ocean tiles
        ocean_mask = ~self.land_mask
        if np.any(ocean_mask):
            min_temp, max_temp = np.min(self.ocean_temp_modifier[ocean_mask]), np.max(self.ocean_temp_modifier[ocean_mask])
            if max_temp > min_temp:
                self.ocean_temp_modifier[ocean_mask] = (self.ocean_temp_modifier[ocean_mask] - min_temp) / (max_temp - min_temp)
            self.ocean_temp_modifier[ocean_mask] = (self.ocean_temp_modifier[ocean_mask] - 0.5) * 2 # Range -1 to 1

    def _apply_biomes(self, land_mask, land_min, land_max, temp_offset=0.0):
        if self.progress_callback: self.progress_callback(50, "Applying biomes: Calculating moisture...")
        
        y_coords = np.arange(self.y_range)
        latitude_temp_1d = 1.0 - (np.abs(y_coords - self.y_range / 2.0) / (self.y_range / 2.0))
        
        latitude_temp_2d = np.tile(latitude_temp_1d, (self.x_range, 1))

        temp_from_ocean = self.ocean_temp_modifier * 0.2
        base_temperature = np.clip(latitude_temp_2d + temp_from_ocean, 0, 1)

        if (land_max - land_min) != 0:
            altitude_norm = (self.world_map - land_min) / (land_max - land_min)
        else:
            altitude_norm = np.zeros_like(self.world_map, dtype=np.float32)

        altitude_temp_effect = self.params.get('altitude_temp_effect', 0.5)
        temperature = base_temperature - (altitude_norm * altitude_temp_effect)
        temperature = np.clip(temperature - temp_offset, 0, 1)

        base_moisture = self._numba_generate_noise_map(self.x_range, self.y_range, self.moisture_perm, self.moisture_grad3, scale=5.0, octaves=4, persistence=0.5, lacunarity=2.0)
        base_moisture = (base_moisture - np.min(base_moisture)) / (np.max(base_moisture) - np.min(base_moisture))
        wind_map = {'West to East': 0, 'East to West': 1, 'North to South': 2, 'South to North': 3}
        wind_direction_code = wind_map.get(self.params.get('wind_direction', 'West to East'), 0)
        simulated_moisture = _numba_simulate_rainfall(self.world_map, land_mask, wind_direction_code, self.ocean_temp_modifier)
        if np.max(simulated_moisture) > 0: simulated_moisture /= np.max(simulated_moisture)
        moisture = (base_moisture * 0.4) + (simulated_moisture * 0.6)
        if np.max(moisture) > 0: moisture /= np.max(moisture)
        self.moisture_map = moisture # Save for river generation

        if self.progress_callback: self.progress_callback(70, "Applying biomes: Classifying biomes...")
        
        alpine_props = BIOME_DEFINITIONS['alpine_glacier']
        alpine_mask = (altitude_norm > 0.7) & (temperature < alpine_props['temp_range'][1]) & land_mask
        if np.any(alpine_mask):
            base_color, num_shades = alpine_props['idx'], alpine_props.get('shades', 1)
            color_indices = base_color + np.floor(altitude_norm[alpine_mask] * (num_shades - 0.01))
            self.color_map[alpine_mask] = color_indices.astype(np.uint8)

        remaining_land_mask = land_mask & ~alpine_mask
        for name, props in BIOME_DEFINITIONS.items():
            if name in ['ocean', 'glacier', 'alpine_glacier', 'river']: continue
            t_min, t_max = props['temp_range']
            m_min, m_max = props['moist_range']
            biome_mask = (temperature >= t_min) & (temperature < t_max) & (moisture >= m_min) & (moisture < m_max) & remaining_land_mask
            if np.any(biome_mask):
                base_color, num_shades = props['idx'], props.get('shades', 1)
                color_indices = base_color + np.floor(altitude_norm[biome_mask] * (num_shades - 0.01))
                self.color_map[biome_mask] = color_indices.astype(np.uint8)
        
        uncolored_land_mask = (self.color_map == 0) & remaining_land_mask
        if np.any(uncolored_land_mask):
            tundra_props = BIOME_DEFINITIONS['tundra']
            base_color, num_shades = tundra_props['idx'], tundra_props.get('shades', 1)
            color_indices = base_color + np.floor(altitude_norm[uncolored_land_mask] * (num_shades - 0.01))
            self.color_map[uncolored_land_mask] = color_indices.astype(np.uint8)

    def _apply_ice_caps(self, percent_ice):
        if percent_ice <= 0: return
        if self.progress_callback: self.progress_callback(80, "Applying ice caps...")
        noise_map = self._numba_generate_noise_map(self.x_range, self.y_range, self.ice_perm, self.ice_grad3, scale=4.0, octaves=6, persistence=0.5, lacunarity=2.0)
        noise_map = (noise_map - np.min(noise_map)) / (np.max(noise_map) - np.min(noise_map))
        y_coords = np.arange(self.y_range)
        latitude_factor = np.tile(np.abs(y_coords - (self.y_range - 1) / 2.0) / (self.y_range / 2.0), (self.x_range, 1))
        ice_score_map = (1.2 * latitude_factor) + (0.4 * noise_map)
        ice_threshold = np.percentile(ice_score_map, 100 - percent_ice)
        ice_mask = ice_score_map >= ice_threshold
        if not np.any(ice_mask): return
        ice_altitudes = self.world_map[ice_mask]
        min_ice_alt, max_ice_alt = np.min(self.world_map), np.max(self.world_map)
        if max_ice_alt == min_ice_alt: max_ice_alt += 1
        ice_base_color, ice_num_shades = BIOME_DEFINITIONS['glacier']['idx'], BIOME_DEFINITIONS['glacier']['shades']
        normalized_ice_alts = (ice_altitudes - min_ice_alt) / (max_ice_alt - min_ice_alt)
        final_ice_colors = ice_base_color + np.floor(normalized_ice_alts * (ice_num_shades - 0.01))
        self.color_map[ice_mask] = final_ice_colors.astype(np.uint8)

    def generate(self):
        self.world_map = self._generate_fractal_height_map()
        
        self._apply_erosion(self.params.get('erosion'))
        
        season = self.params.get('season', 'Normal')
        season_offsets = {'Normal': 0.0, 'Warm': 0.1, 'Cold': -0.1, 'Ice Age': -0.25}
        temp_offset = season_offsets.get(season, 0.0)
        
        self.finalize_map(self.params['water'], self.params.get('ice', 15.0), self.params.get('map_style', 'Biome'), temp_offset)
        
        self._generate_rivers()
        
        self.used_names, self.major_features = set(), []
        self.natural_features = self.generate_natural_features()
        self.generate_settlements()
        
        if self.progress_callback: self.progress_callback(99, "Creating image...")
        self.pil_image = self.create_image()
        if self.progress_callback: self.progress_callback(100, "Done.")
        return self.pil_image

    def finalize_map(self, percent_water, percent_ice, map_style, temp_offset=0.0):
        if self.progress_callback: self.progress_callback(35, "Finalizing map...")
        min_z, max_z = np.min(self.world_map), np.max(self.world_map)
        if max_z == min_z: max_z += 1
        
        water_level_threshold = np.percentile(self.world_map, percent_water)
        
        self.color_map = np.zeros_like(self.world_map, dtype=np.uint8)
        water_mask = self.world_map < water_level_threshold
        self.land_mask = ~water_mask
        
        self._simulate_ocean_currents()

        water_min, water_max = min_z, water_level_threshold
        if water_max == water_min: water_max += 1
        if (water_max - water_min) != 0:
            self.color_map[water_mask] = 1 + np.floor(6.99 * (self.world_map[water_mask] - water_min) / (water_max - water_min))
        else:
            self.color_map[water_mask] = 1

        land_min, land_max = water_level_threshold, max_z
        if land_max == land_min: land_max += 1
        
        if map_style == 'Biome':
            self._apply_biomes(self.land_mask, land_min, land_max, temp_offset)
        else:
            if (land_max - land_min) != 0:
                normalized_altitude = (self.world_map[self.land_mask] - land_min) / (land_max - land_min)
                self.color_map[self.land_mask] = 16 + np.floor(15.99 * normalized_altitude)
            else:
                self.color_map[self.land_mask] = 16

        self.color_map_before_ice = self.color_map.copy()
        self._apply_ice_caps(percent_ice + (20 if self.params.get('season') == 'Ice Age' else 0))

    def _generate_rivers(self):
        if self.progress_callback: self.progress_callback(90, "Generating rivers and lakes...")
        if not hasattr(self, 'moisture_map') or not hasattr(self, 'land_mask') or not np.any(self.land_mask):
            self.rivers = []
            return

        # --- 1. Fill depressions to create lakes and a routable height map ---
        height_map_float = self.world_map.astype(np.float32)
        
        pq = [] # Priority queue with (elevation, x, y)
        
        for x in range(self.x_range):
            for y in range(self.y_range):
                if not self.land_mask[x, y] or x == 0 or x == self.x_range - 1 or y == 0 or y == self.y_range - 1:
                    heapq.heappush(pq, (height_map_float[x, y], x, y))

        filled_map = np.full_like(height_map_float, np.inf)
        for _, x, y in pq:
            filled_map[x, y] = height_map_float[x, y]

        while pq:
            h, x, y = heapq.heappop(pq)
            if h > filled_map[x, y]:
                continue
            
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = (x + dx + self.x_range) % self.x_range, y + dy
                if 0 <= ny < self.y_range:
                    new_h = max(h, height_map_float[nx, ny])
                    if new_h < filled_map[nx, ny]:
                        filled_map[nx, ny] = new_h
                        heapq.heappush(pq, (new_h, nx, ny))
        
        # --- 2. Identify and selectively draw lakes based on coverage ---
        lake_coverage = self.params.get('lake_coverage', 50.0)
        potential_lake_mask = (filled_map > height_map_float) & self.land_mask
        
        # --- Prevent lakes in deserts ---
        desert_props = BIOME_DEFINITIONS['desert']
        desert_mask = (self.color_map >= desert_props['idx']) & (self.color_map < desert_props['idx'] + desert_props.get('shades', 1))
        potential_lake_mask[desert_mask] = False
        
        if np.any(potential_lake_mask) and lake_coverage > 0:
            labeled_lakes, num_lakes = label(potential_lake_mask)
            if num_lakes > 0:
                lake_sizes = [(l, np.sum(labeled_lakes == l)) for l in range(1, num_lakes + 1)]
                lake_sizes.sort(key=lambda item: item[1], reverse=True)
                
                num_to_keep = int(num_lakes * (lake_coverage / 100.0))
                labels_to_keep = {l[0] for l in lake_sizes[:num_to_keep]}
                
                final_lake_mask = np.isin(labeled_lakes, list(labels_to_keep))
                
                self.land_mask[final_lake_mask] = False
                self.color_map[final_lake_mask] = 4 
                self.world_map[final_lake_mask] = filled_map[final_lake_mask].astype(np.int32)

        # --- 3. Generate rivers on the depression-less landscape ---
        num_rivers = self.params.get('num_rivers', 50)
        self.rivers = []
        river_map = np.zeros_like(self.world_map, dtype=np.float32)
        
        # Diversified river origins: lower the percentile requirement
        potential_sources = (self.world_map > np.percentile(self.world_map[self.land_mask], 75)) & \
                            (self.moisture_map > np.percentile(self.moisture_map[self.land_mask], 75)) & \
                            self.land_mask
        
        source_coords = np.argwhere(potential_sources)
        if source_coords.shape[0] == 0: return

        for _ in range(num_rivers):
            if source_coords.shape[0] == 0: break
            start_index = random.randint(0, source_coords.shape[0] - 1)
            x, y = source_coords[start_index]
            
            path = []
            flow = 1.0
            
            for _ in range(self.x_range): # Max river length
                path.append((x, y))
                
                if not self.land_mask[x, y]: break 

                min_h = filled_map[x, y]
                next_pos = None
                
                neighbors = [((x+dx+self.x_range)%self.x_range, y+dy) for dx, dy in [(-1,0),(1,0),(0,-1),(0,1), (-1,-1), (1,-1), (-1,1), (1,1)]]
                random.shuffle(neighbors)

                for nx, ny in neighbors:
                    if 0 <= ny < self.y_range and filled_map[nx, ny] < min_h:
                        min_h = filled_map[nx, ny]
                        next_pos = (nx, ny)
                
                if next_pos:
                    self.world_map[x, y] = max(self.world_map[x, y] - 2, int(min_h))
                    river_map[x, y] += flow
                    x, y = next_pos
                    flow += 0.2
                else:
                    break
            
            if len(path) > 5:
                self.rivers.append({'path': path, 'flow': flow})

        # --- 4. Draw rivers on the map ---
        river_color = BIOME_DEFINITIONS['river']['idx']
        river_mask = river_map > 0
        self.color_map[river_mask] = river_color

    def generate_settlements(self):
        num_settlements = self.params['num_settlements']
        if num_settlements == 0: self.settlements = []; return
        if self.progress_callback: self.progress_callback(98, "Placing settlements...")
        
        suitability = np.zeros(self.world_map.shape, dtype=np.float32)
        for name, score in {'temperate_forest':1.0, 'temperate_rainforest':0.9, 'shrubland':0.8, 'savanna':0.7, 'tropical_forest':0.6, 'tropical_rainforest':0.5, 'taiga':0.4, 'desert':0.2, 'tundra':0.1}.items():
            props = BIOME_DEFINITIONS[name]
            suitability[(self.color_map >= props['idx']) & (self.color_map < props['idx'] + props.get('shades', 1))] = score
        
        y_coords = np.arange(self.y_range)
        latitude_score = 1.0 - (np.abs(y_coords - self.y_range/2.0)/(self.y_range/2.0))
        suitability *= latitude_score[np.newaxis, :]**2
        
        if self.rivers:
            river_mask = np.zeros_like(self.world_map, dtype=bool)
            for river in self.rivers:
                for x, y in river['path']:
                    river_mask[x, y] = True
            dist_to_river = distance_transform_edt(~river_mask)
            river_bonus = 2.0 / (dist_to_river**0.5 + 1)
            suitability[self.land_mask] += river_bonus[self.land_mask]

        distance_to_water = distance_transform_edt(self.land_mask)
        water_bonus = 0.8 / (distance_to_water**0.5 + 1e-6)
        suitability[self.land_mask] += water_bonus[self.land_mask]

        grad_x, grad_y = np.gradient(self.world_map.astype(np.float32))
        slope = np.sqrt(grad_x**2 + grad_y**2)
        if np.max(slope) > 0:
            suitability[self.land_mask] -= (slope[self.land_mask] / np.max(slope)) * 1.5
        
        settlement_noise_map = self._numba_generate_noise_map(self.x_range, self.y_range, self.settlement_perm, self.settlement_grad3, scale=20.0, octaves=4, persistence=0.5, lacunarity=2.0)
        suitability[self.land_mask] += ((settlement_noise_map[self.land_mask] + 1.0) / 2.0) * 0.2
        suitability[suitability < 0] = 0

        flat_suitability = suitability.flatten()
        possible_indices = np.where(flat_suitability > 0)[0]
        if possible_indices.size == 0: self.settlements = []; return
        num_candidates = min(possible_indices.size, num_settlements * 20)
        best_candidate_indices = possible_indices[np.argpartition(flat_suitability[possible_indices], -num_candidates)[-num_candidates:]]
        min_dist_sq = (math.sqrt(self.x_range * self.y_range) / (num_settlements * 0.5 + 10))**2
        
        self.settlements = []
        sorted_candidates = best_candidate_indices[np.argsort(flat_suitability[best_candidate_indices])][::-1]
        name_fragments = THEME_NAME_FRAGMENTS.get(self.params.get('theme', 'High Fantasy'), THEME_NAME_FRAGMENTS['High Fantasy'])
        
        for idx in sorted_candidates:
            if len(self.settlements) >= num_settlements: break
            x, y = np.unravel_index(idx, (self.x_range, self.y_range))
            is_far_enough = all(min(abs(x-s['x']), self.x_range - abs(x-s['x']))**2 + (y-s['y'])**2 > min_dist_sq for s in self.settlements)
            if is_far_enough:
                context = self._find_nearest_major_feature(x, y)
                name = generate_contextual_name(name_fragments, self.used_names, context)
                stype = random.choice(name_fragments['types'])
                self.settlements.append({'x': x, 'y': y, 'name': name, 'type': 'settlement', 'stype': stype})

    def _find_nearest_major_feature(self, x, y):
        if not self.major_features: return None
        min_dist_sq, closest_feature = float('inf'), None
        for feature in self.major_features:
            center_x, center_y = feature['center']
            dx = min(abs(x - center_x), self.x_range - abs(x - center_x))
            dist_sq = dx**2 + (y - center_y)**2
            if dist_sq < min_dist_sq:
                min_dist_sq, closest_feature = dist_sq, feature
        if closest_feature and min_dist_sq < (self.x_range / 5)**2:
             return closest_feature
        return None

    def _find_features(self, mask, min_size=50):
        features = []
        labeled_array, num_features = label(mask)
        if num_features > 0:
            slices = find_objects(labeled_array)
            for i, slc in enumerate(slices):
                if slc is None: continue
                feature_mask = labeled_array[slc] == i + 1
                size = np.sum(feature_mask)
                if size >= min_size:
                    coords = np.argwhere(feature_mask)
                    mean_local_x, mean_local_y = np.mean(coords, axis=0)
                    center_x = int(mean_local_x + slc[0].start)
                    center_y = int(mean_local_y + slc[1].start)
                    features.append({'size': int(size), 'center': (center_x, center_y), 'label': i + 1})
        features.sort(key=lambda f: f['size'], reverse=True)
        return features, labeled_array

    def generate_natural_features(self):
        if self.world_map is None: return {}
        if self.progress_callback: self.progress_callback(97, "Naming natural features...")
        num_features = self.params.get('num_features', 10)
        theme = self.params.get('theme', 'High Fantasy')
        name_fragments = THEME_NAME_FRAGMENTS.get(theme, THEME_NAME_FRAGMENTS['High Fantasy'])
        features = {'peaks': [], 'ranges': [], 'areas': []}
        if not np.any(self.land_mask): return features

        all_candidates = []
        min_land_feature_size = max(50, int(np.sum(self.land_mask) * 0.01))
        
        water_bodies, self.labeled_water = self._find_features(~self.land_mask, min_size=1)
        
        edge_labels = set(self.labeled_water[0, :]) | set(self.labeled_water[-1, :])
        edge_labels.discard(0) 

        edge_water = []
        inland_water = []

        for body in water_bodies:
            if body['label'] in edge_labels:
                edge_water.append(body)
            else:
                inland_water.append(body)
        
        if edge_water:
            edge_water.sort(key=lambda x: x['size'], reverse=True)
            all_candidates.append({'feature': edge_water.pop(0), 'type': 'ocean'})
            for body in edge_water:
                if body['size'] > int(self.x_range * self.y_range * 0.002):
                    all_candidates.append({'feature': body, 'type': 'sea'})

        # Control how many lakes get named
        max_lakes_to_name = self.params.get('max_lakes_to_name', 20)
        lakes_to_name = inland_water[:max_lakes_to_name]
        for body in lakes_to_name:
            all_candidates.append({'feature': body, 'type': 'lake'})

        min_mountain_height = np.percentile(self.world_map[self.land_mask], 90)
        mountain_mask = (self.world_map >= min_mountain_height) & self.land_mask
        labeled_mountains, num_mountain_features = label(mountain_mask)
        if num_mountain_features > 0:
            slices = find_objects(labeled_mountains)
            for i, slc in enumerate(slices):
                if slc is None: continue
                feature_mask = labeled_mountains[slc] == i + 1
                if np.sum(feature_mask) >= min_land_feature_size:
                    coords = np.argwhere(feature_mask)
                    mean_local_x, mean_local_y = np.mean(coords, axis=0)
                    center = (int(mean_local_x + slc[0].start), int(mean_local_y + slc[1].start))
                    all_candidates.append({'feature': {'size': np.sum(feature_mask), 'center': center}, 'type': 'range'})
                    
                    heights_in_slice = self.world_map[slc]
                    heights_in_feature = np.where(feature_mask, heights_in_slice, -np.inf)
                    local_peak_coords = np.unravel_index(np.argmax(heights_in_feature), heights_in_feature.shape)
                    peak_coord_x = local_peak_coords[0] + slc[0].start
                    peak_coord_y = local_peak_coords[1] + slc[1].start
                    peak_coord = (peak_coord_x, peak_coord_y)

                    all_candidates.append({'feature': {'size': self.world_map[peak_coord], 'x': peak_coord[0], 'y': peak_coord[1]}, 'type': 'peak'})

        for biome_key, f_type in {'desert':'desert', 'tropical_rainforest':'jungle', 'temperate_forest':'forest', 'tundra':'wastes'}.items():
            props = BIOME_DEFINITIONS[biome_key]
            mask = (self.color_map_before_ice >= props['idx']) & (self.color_map_before_ice < props['idx'] + props.get('shades', 1))
            found_features, _ = self._find_features(mask, min_size=min_land_feature_size)
            for feature in found_features:
                 all_candidates.append({'feature': feature, 'type': f_type})
        
        all_candidates.sort(key=lambda x: x['feature']['size'], reverse=True)
        
        major_feature_types = {'ocean', 'range', 'desert', 'jungle', 'forest'}
        temp_candidates = list(all_candidates)
        for cand in temp_candidates:
            if len(self.major_features) >= 5: break
            if cand['type'] in major_feature_types:
                base_name = generate_fantasy_name(name_fragments, self.used_names)
                name = f"The {base_name} {cand['type'].capitalize()}"
                
                final_feature = cand['feature']
                final_feature.update({'name': name, 'type': cand['type']})
                self.major_features.append(final_feature)
                if cand['type'] == 'range': features['ranges'].append(final_feature)
                else: features['areas'].append(final_feature)
                all_candidates.remove(cand)

        named_count = len(self.major_features)
        for cand in all_candidates:
            if named_count >= num_features: break
            base_name = generate_fantasy_name(name_fragments, self.used_names)
            name_map = {'sea': f"The {base_name} Sea", 'peak': f"Mount {base_name}", 'lake': f"Lake {base_name}"}
            name = name_map.get(cand['type'], base_name)
            
            final_feature = cand['feature']
            final_feature.update({'name': name, 'type': cand['type']})
            
            if cand['type'] == 'peak': features['peaks'].append(final_feature)
            else: features['areas'].append(final_feature)
            named_count += 1
                
        return features

    def create_image(self):
        if self.color_map is None: return None
        clipped_color_map = np.clip(self.color_map, 0, len(self.palette) - 1)
        
        palette_array = np.array(self.palette, dtype=np.uint8)
        rgb_map = palette_array[clipped_color_map.T]

        return Image.fromarray(rgb_map, 'RGB')
