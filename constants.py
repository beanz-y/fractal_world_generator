# beanz-y/fractal_world_generator/fractal_world_generator-28f75751b57dacf83432892d2293f1e3754a3ba6/constants.py
#
# --- CHANGELOG ---
# 1. Configuration Management:
#    - Removed hardcoded BIOME_DEFINITIONS and THEME_NAME_FRAGMENTS dictionaries.
#    - Added a helper function to load data from external JSON files.
#    - BIOME_DEFINITIONS are now loaded from 'biomes.json'.
#    - THEME_NAME_FRAGMENTS are now loaded from 'themes.json'.
#    - This allows for easier user customization without editing source code.
# -----------------

import json
import os

def _load_json_from_file(filename):
    """
    Helper function to load data from a JSON file located in the same
    directory as this script. This makes the path relative and robust.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Provide a more helpful error message if the files are missing.
        raise FileNotFoundError(
            f"Error: The configuration file '{filename}' was not found. "
            f"Please make sure '{filename}' is in the same directory as 'constants.py'."
        )
    except json.JSONDecodeError as e:
        # Provide context for JSON errors.
        raise json.JSONDecodeError(
            f"Error decoding '{filename}': {e.msg}", e.doc, e.pos
        )

# Load the definitions from the external files at startup.
BIOME_DEFINITIONS = _load_json_from_file('biomes.json')
THEME_NAME_FRAGMENTS = _load_json_from_file('themes.json')


# PREDEFINED_PALETTES remains here as it's fundamental to the rendering
# logic and less likely to be customized by a casual user.
PREDEFINED_PALETTES = {
    "Biome": [
        # 0: Unused
        (0,0,0), 
        # 1-7: Water
        (28,82,106), (43,105,128), (57,128,149), (72,150,171), (86,173,192), (101,196,214), (115,219,235),
        # 8-15: Unused
        (0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),
        # 16-23: Desert
        (249,225,184), (244,215,165), (239,205,146), (234,195,127), (229,185,108), (224,175,89), (219,165,70), (214,155,51),
        # 24-31: Savanna / Shrubland
        (185,209,139), (170,198,121), (155,187,103), (140,176,85), (125,165,67), (110,154,49), (95,143,31), (80,132,13),
        # 32-39: Forest
        (134,188,128), (119,173,113), (104,158,98), (89,143,83), (74,128,68), (59,113,53), (44,98,38), (29,83,23),
        # 40-47: Taiga
        (180,191,170), (171,181,162), (162,171,153), (153,161,145), (144,151,136), (135,141,128), (126,131,119), (117,121,111),
        # 48-51: Tundra / Rock
        (136,136,136), (150,150,150), (164,164,164), (178,178,178),
        # 52-55: Glacier / Polar Ice
        (255,255,255), (245,245,245), (235,235,235), (225,225,225),
        # 56-59: Alpine Glacier
        (220, 230, 240), (210, 220, 230), (200, 210, 220), (190, 200, 210),
        # 60: River
        (74, 127, 221),
        # 61-63: Unused
        (0,0,0),(0,0,0),(0,0,0),
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
