# beanz-y/fractal_world_generator/fractal_world_generator-8b752999818ebdee7e3c696935b618f2a364ff8f/projection.py
import numpy as np
import math

def render_globe(source_array, canvas_w, canvas_h, rot_y_deg, rot_x_deg, x_range, y_range):
    """Renders an equirectangular map array onto a globe projection."""
    rot_y, rot_x = math.radians(rot_y_deg), math.radians(rot_x_deg)
    cy, sy, cx, sx = math.cos(rot_y), math.sin(rot_y), math.cos(rot_x), math.sin(rot_x)
    
    # Create a grid of coordinates for the canvas
    canvas_coords_x, canvas_coords_y = np.meshgrid(np.arange(canvas_w), np.arange(canvas_h))
    radius = min(canvas_w, canvas_h) / 2
    
    # Convert canvas coordinates to Normalized Device Coordinates (-1 to 1)
    ndc_x = (canvas_coords_x - canvas_w / 2) / radius
    ndc_y = (canvas_coords_y - canvas_h / 2) / radius
    
    # Calculate Z coordinate for each point on the sphere, and create a mask for visible points
    z2 = 1 - ndc_x**2 - ndc_y**2
    visible_mask = z2 >= 0
    z = np.sqrt(z2[visible_mask])
    x_ndc_masked, y_ndc_masked = ndc_x[visible_mask], ndc_y[visible_mask]
    
    # Apply inverse rotations to find the corresponding point on the un-rotated sphere
    # Inverse pitch (around X-axis)
    y_r1 = y_ndc_masked * cx + z * sx
    z_r1 = -y_ndc_masked * sx + z * cx
    x_r1 = x_ndc_masked

    # Inverse yaw (around Y-axis)
    x_r2 = x_r1 * cy + z_r1 * sy
    z_r2 = -x_r1 * sy + z_r1 * cy
    y_r2 = y_r1

    # Convert 3D cartesian coordinates to spherical (latitude and longitude)
    lat, lon = np.arcsin(np.clip(y_r2, -1, 1)), np.arctan2(x_r2, z_r2)
    
    # Convert latitude and longitude to source image pixel coordinates
    # --- CHANGE START ---
    # Inverted src_y calculation to flip the map on the globe
    src_y = np.clip(((lat / math.pi) + 0.5) * y_range, 0, y_range - 1)
    # --- CHANGE END ---
    src_x = np.clip(((lon / math.pi) * 0.5 + 0.5) * x_range, 0, x_range - 1)
    
    # Create the new image array and populate it with pixels from the source
    new_img_array = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    new_img_array[canvas_coords_y[visible_mask], canvas_coords_x[visible_mask]] = source_array[src_y.astype(int), src_x.astype(int)]
    
    return new_img_array

def canvas_coords_to_map(canvas_x, canvas_y, canvas_w, canvas_h, rot_y_deg, rot_x_deg, x_range, y_range):
    """Converts a coordinate on the canvas to a coordinate on the original map image."""
    rot_y, rot_x = math.radians(rot_y_deg), math.radians(rot_x_deg)
    cy, sy, cx, sx = math.cos(rot_y), math.sin(rot_y), math.cos(rot_x), math.sin(rot_x)
    radius = min(canvas_w, canvas_h) / 2
    if radius == 0: return None, None
    
    # Convert canvas coordinates to Normalized Device Coordinates
    xs = (canvas_x - canvas_w / 2) / radius
    ys = -(canvas_y - canvas_h / 2) / radius
    
    # Check if the click is on the globe, and find its 3D position
    z2 = 1 - xs**2 - ys**2
    if z2 < 0: return None, None
    zs = math.sqrt(z2)

    # Apply inverse rotations (Pitch then Yaw)
    # Inverse Pitch
    y1 = ys * cx + zs * sx
    z1 = -ys * sx + zs * cx
    
    # Inverse Yaw (Corrected)
    x0 = xs * cy + z1 * sy
    z0 = -xs * sy + z1 * cy
    y0 = y1
    
    # Convert 3D cartesian coordinates to spherical (lat/lon)
    lat = math.asin(np.clip(y0, -1, 1))
    lon = math.atan2(x0, z0)
    
    # Convert lat/lon to map pixel coordinates
    # This now corresponds to the inverted mapping in render_globe
    map_y = int(((lat / math.pi) + 0.5) * y_range)
    map_x = int(((lon / math.pi) * 0.5 + 0.5) * x_range)
    return map_x, map_y

def map_coords_to_canvas(img_x, img_y, canvas_w, canvas_h, rot_y_deg, rot_x_deg, x_range, y_range):
    """Converts a coordinate on the original map image to a coordinate on the canvas globe."""
    # Convert map pixel coordinates to lat/lon
    # --- CHANGE START ---
    # Inverted latitude calculation to flip the map
    lat = (img_y / y_range - 0.5) * math.pi
    # --- CHANGE END ---
    lon = (img_x / x_range - 0.5) * 2 * math.pi

    # Convert lat/lon to 3D cartesian coordinates
    x3d = math.cos(lat) * math.cos(lon)
    y3d = math.sin(lat)
    z3d = math.cos(lat) * math.sin(lon)

    rot_y, rot_x = math.radians(rot_y_deg), math.radians(rot_x_deg)
    cy, sy, cx, sx = math.cos(rot_y), math.sin(rot_y), math.cos(rot_x), math.sin(rot_x)

    # Apply forward rotations (Yaw then Pitch)
    # Yaw
    x_r1 = x3d * cy - z3d * sy
    z_r1_pre_pitch = x3d * sy + z3d * cy
    
    # Pitch
    y_r2 = y3d * cx - z_r1_pre_pitch * sx
    z_r2 = y3d * sx + z_r1_pre_pitch * cx

    # Backface culling: don't draw points on the far side of the globe
    if z_r2 < 0:
        return None, None

    # Project the 3D point onto the 2D canvas
    radius = min(canvas_w, canvas_h) / 2
    canvas_x = x_r1 * radius + canvas_w / 2
    canvas_y = -y_r2 * radius + canvas_h / 2
    
    return canvas_x, canvas_y