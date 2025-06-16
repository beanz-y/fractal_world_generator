# beanz-y/fractal_world_generator/fractal_world_generator-8b752999818ebdee7e3c696935b618f2a364ff8f/world_generator.py
import numpy as np # type: ignore
import random
import math
from PIL import Image # type: ignore
from utils import SimplexNoise
from constants import BIOME_DEFINITIONS, PREDEFINED_PALETTES, THEME_NAME_FRAGMENTS
import numba
from collections import deque

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

    # Note: The 'parallel=True' argument was removed because the simulation loop
    # has a sequential dependency (each step depends on the last), making it
    # unsuitable for parallel execution. Numba still JIT-compiles this function
    # for a significant speedup.
    for i in range(y_range if wind_direction < 2 else x_range):
        air_moisture = 1.0

        # WARM-UP PASS for horizontal winds to ensure seamless wrapping
        if wind_direction < 2:
            for j_iter in range(x_range):
                if wind_direction == 0: # West to East
                    x, y, px, py = j_iter, i, (j_iter - 1 + x_range) % x_range, i
                else: # East to West
                    x, y, px, py = (x_range - 1 - j_iter), i, (x_range - 1 - ((j_iter - 1 + x_range) % x_range)), i

                if not land_mask[x, y]:
                    air_moisture = min(1.0, air_moisture + 0.05)
                    continue

                air_moisture += 0.002
                elevation_diff = height_map[x, y] - height_map[px, py]

                precipitation = 0.015 * air_moisture
                if elevation_diff > 0:
                    precipitation += elevation_diff * 12.0 * air_moisture * 0.001
                
                precipitation = min(air_moisture, precipitation)
                # No precipitation is recorded here, just updating air moisture
                air_moisture = max(0, air_moisture - precipitation)
        
        # RECORDING PASS for all wind directions
        loop_len = x_range if wind_direction < 2 else y_range
        for j_iter in range(loop_len):
            # Determine coordinates based on wind direction
            if wind_direction == 0: # West to East
                x, y, px, py = j_iter, i, (j_iter - 1 + x_range) % x_range, i
            elif wind_direction == 1: # East to West
                x, y, px, py = (x_range - 1 - j_iter), i, (x_range - 1 - ((j_iter - 1 + x_range) % x_range)), i
            elif wind_direction == 2: # North to South
                x, y, px, py = i, j_iter, i, j_iter - 1
                if j_iter == 0: px, py = x, y # No previous pixel
            else: # South to North
                x, y, px, py = i, (y_range - 1 - j_iter), i, (y_range - j_iter)
                if j_iter == 0: px, py = x, y # No previous pixel

            if not land_mask[x, y]:
                air_moisture = min(1.0, air_moisture + 0.05)
                continue

            air_moisture += 0.002
            elevation_diff = height_map[x, y] - height_map[px, py]

            precipitation = 0.015 * air_moisture
            if elevation_diff > 0:
                precipitation += elevation_diff * 12.0 * air_moisture * 0.001
            
            precipitation = min(air_moisture, precipitation)
            moisture_map[x, y] = precipitation # Set precipitation, not accumulate
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
        for i in range(erosion_passes):
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
        
        # Reverted to original heightmap generation logic
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
        # FIX: Apply water coloring to the correct mask (water_mask)
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
        using a Breadth-First Search (BFS) algorithm.
        """
        if self.progress_callback: self.progress_callback(96, "Placing settlements: Calculating water distance...")
        
        x_range, y_range = self.x_range, self.y_range
        distance_map = np.full(land_mask.shape, np.inf, dtype=np.float32)
        queue = deque()

        # Efficiently find all land pixels adjacent to water (the "coast") to seed the BFS
        water_mask = ~land_mask
        coast_mask = (
            (np.roll(water_mask, 1, axis=0) & land_mask) |
            (np.roll(water_mask, -1, axis=0) & land_mask) |
            (np.roll(water_mask, 1, axis=1) & land_mask) |
            (np.roll(water_mask, -1, axis=1) & land_mask)
        )
        
        # Initialize the queue with all coastal pixels at distance 1
        coast_pixels = np.argwhere(coast_mask)
        for x, y in coast_pixels:
            distance_map[x, y] = 1
            queue.append(((x, y), 1))

        # Perform BFS from the coast
        visited = set(map(tuple, coast_pixels))
        
        while queue:
            (x, y), dist = queue.popleft()
            
            # Check neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = (x + dx), y + dy # Horizontal wrapping is handled later if needed

                # Wrap horizontally for seamless world
                nx = nx % x_range
                
                if 0 <= ny < y_range and land_mask[nx, ny] and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    distance_map[nx, ny] = dist + 1
                    queue.append(((nx, ny), dist + 1))
        
        return distance_map

    def generate_settlements(self):
        """
        Generates settlement locations based on a more realistic suitability model,
        considering biome type, proximity to water, altitude, and slope.
        """
        if self.world_map is None or self.land_mask is None:
            return

        num_settlements = self.params['num_settlements']
        theme = self.params.get('theme', 'High Fantasy')
        if num_settlements == 0:
            self.settlements = []
            return

        if self.progress_callback:
            self.progress_callback(96, "Placing settlements: Calculating suitability...")
            
        # 1. Start with a base suitability score determined by the biome.
        BIOME_SUITABILITY = {
            'temperate_forest': 1.0, 'temperate_rainforest': 0.9,
            'shrubland': 0.8, 'savanna': 0.7,
            'tropical_forest': 0.6, 'tropical_rainforest': 0.5,
            'taiga': 0.4, 'desert': 0.2, 'tundra': 0.1
        }
        
        # Create a map of suitability scores based on the biome of each pixel
        suitability = np.full(self.world_map.shape, -1.0, dtype=np.float32)
        for name, props in BIOME_DEFINITIONS.items():
            if name in BIOME_SUITABILITY:
                # Get all color indexes for this biome
                start_idx, end_idx = props['idx'], props['idx'] + props.get('shades', 1)
                biome_mask = (self.color_map >= start_idx) & (self.color_map < end_idx)
                suitability[biome_mask] = BIOME_SUITABILITY[name]

        # 2. Add a strong bonus for proximity to water.
        if np.any(self.land_mask):
            distance_to_water = self._calculate_distance_to_water(self.land_mask)
            # Apply a bonus, diminishing with distance. Max bonus of 0.8 for coastlines.
            water_bonus_mask = np.isfinite(distance_to_water)
            suitability[water_bonus_mask] += 0.8 / (distance_to_water[water_bonus_mask] ** 0.5)

        # Create a mask for all valid settlement locations (anywhere with a base score)
        valid_land = suitability > 0
        
        # 3. Penalize high altitude on valid land
        land_altitudes = self.world_map[valid_land]
        if land_altitudes.size > 0:
            min_land_alt, max_land_alt = np.min(land_altitudes), np.max(land_altitudes)
            if max_land_alt > min_land_alt:
                normalized_land_alt = (self.world_map[valid_land] - min_land_alt) / (max_land_alt - min_land_alt)
                suitability[valid_land] -= normalized_land_alt * 0.7 # Increased altitude penalty

        # 4. Penalize steep slopes on valid land
        grad_x, grad_y = np.gradient(self.world_map.astype(np.float32))
        slope = np.sqrt(grad_x**2 + grad_y**2)
        if np.max(slope) > 0:
            normalized_slope = slope / np.max(slope)
            suitability[valid_land] -= normalized_slope[valid_land] * 1.5 # Adjusted slope penalty

        # 5. Add noise for variety, only to valid land
        settlement_noise_map = self._numba_generate_noise_map(
            self.x_range, self.y_range, self.settlement_perm, self.settlement_grad3,
            scale=20.0, octaves=4, persistence=0.5, lacunarity=2.0
        )
        suitability[valid_land] += ((settlement_noise_map[valid_land] + 1.0) / 2.0) * 0.2 # Reduced noise influence

        # Any location that fell below zero suitability is now invalid.
        suitability[suitability < 0] = -9999.0

        # 6. Find the best locations from the final suitability map
        flat_suitability = suitability.flatten()
        num_candidates = min(len(flat_suitability[flat_suitability > -9999.0]), num_settlements * 20)
        if num_candidates <= 0:
            self.settlements = []
            return

        candidate_indices = np.argpartition(flat_suitability, -num_candidates)[-num_candidates:]
        
        min_dist = math.sqrt(self.x_range * self.y_range) / (num_settlements * 0.5 + 10)
        min_dist_sq = min_dist**2

        self.settlements = []
        
        # Sort candidates by suitability
        sorted_candidates = candidate_indices[np.argsort(flat_suitability[candidate_indices])][::-1]

        name_fragments = THEME_NAME_FRAGMENTS.get(theme, THEME_NAME_FRAGMENTS['High Fantasy'])
        
        if self.progress_callback: self.progress_callback(98, "Placing settlements: Finalizing locations...")

        for idx in sorted_candidates:
            if len(self.settlements) >= num_settlements:
                break

            x, y = np.unravel_index(idx, (self.x_range, self.y_range))
            
            is_far_enough = True
            for existing_settlement in self.settlements:
                ex, ey = existing_settlement['x'], existing_settlement['y']
                # Account for horizontal wrapping when checking distance
                dx = abs(x - ex)
                if dx > self.x_range / 2:
                    dx = self.x_range - dx
                dist_sq = dx**2 + (y - ey)**2

                if dist_sq < min_dist_sq:
                    is_far_enough = False
                    break
            
            if is_far_enough:
                name = random.choice(name_fragments['prefixes']) + random.choice(name_fragments['suffixes'])
                settlement_type = random.choice(name_fragments['types'])
                self.settlements.append({'x': x, 'y': y, 'name': name, 'type': settlement_type})
    
    def create_image(self):
        if self.color_map is None: return None
        
        clipped_color_map = np.clip(self.color_map, 0, len(self.palette) - 1)
        
        rgb_map = np.zeros((self.y_range, self.x_range, 3), dtype=np.uint8)
        for i, color in enumerate(self.palette):
            mask = clipped_color_map.T == i
            rgb_map[mask] = color
            
        return Image.fromarray(rgb_map, 'RGB')