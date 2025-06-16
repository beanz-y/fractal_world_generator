# beanz-y/fractal_world_generator/fractal_world_generator-8b752999818ebdee7e3c696935b618f2a364ff8f/constants.py
BIOME_DEFINITIONS = {
    'glacier':              {'idx': 52, 'shades': 4, 'temp_range': (0.0, 0.125), 'moist_range': (0.0, 1.0)},
    'alpine_glacier':       {'idx': 56, 'shades': 4, 'temp_range': (0.0, 0.15), 'moist_range': (0.0, 1.0)}, # New biome
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
        # New: Alpine Glacier (56-59) - A slightly bluer, icier white
        (220, 230, 240), (210, 220, 230), (200, 210, 220), (190, 200, 210),
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

THEME_NAME_FRAGMENTS = {
    'High Fantasy': {
        'prefixes': ['Ael', 'Bara', 'Cael', 'Dra', 'El', 'Fael', 'Gal', 'Har', 'Ith', 'Kor', 'Luth', 'Mor', 'Norn', 'Oth', 'Pyr', 'Quel', 'Rath', 'Sil', 'Tir', 'Val'],
        'suffixes': ['don', 'dor', 'dras', 'fall', 'fast', 'garde', 'heim', 'ia', 'is', 'kar', 'land', 'lor', 'mar', 'nor', 'os', 'thal', 'thas', 'tine', 'vale', 'wood'],
        'types': ['Castle', 'Keep', 'Village', 'Town', 'City', 'Citadel', 'Hold', 'Fortress']
    },
    'Sci-Fi': {
        'prefixes': ['Alpha', 'Cygnus', 'Helios', 'Hyper', 'Kepler', 'Nexus', 'Orion', 'Plex', 'Stel', 'Terra', 'Tycho', 'Vex', 'Xylo', 'Zenith', 'Andro'],
        'suffixes': ['-7', ' Prime', ' Station', ' IX', ' Colony', ' Base', ' One', ' Omega', ' Terminus', ' Hub', ' Complex', ' Citadel', ' Point', ' Major', ' Minor'],
        'types': ['Outpost', 'Colony', 'Station', 'Base', 'Habitat', 'Complex', 'Dome']
    },
    'Post-Apocalyptic': {
        'prefixes': ['Rust', 'Ash', 'Dust', 'Scrap', 'Bone', 'Wreck', 'Last', 'New', 'Fort', 'Old', 'Sunken', 'Broken', 'Grit'],
        'suffixes': ['-Town', ' Heap', ' Hope', ' Haven', ' Rock', ' Reach', ' Scrap', ' Yard', ' Camp', ' Refuge', ' Point', ' Out', ' Pit', ' Fall'],
        'types': ['Camp', 'Settlement', 'Refuge', 'Fort', 'Scrap-town', 'Holdout']
    },
}