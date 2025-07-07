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
        'prefixes': ['Aer', 'Amn', 'Ard', 'Ast', 'Bel', 'Bór', 'Cael', 'Cal', 'Cel', 'Cor', 'Cyr', 'Dal', 'Dor', 'Dún', 'Ed', 'El', 'Er', 'Est', 'Fael', 'Fen', 'Gal', 'Gîl', 'Gond', 'Har', 'Hel', 'Il', 'Is', 'Ith', 'Kor', 'Lael', 'Lin', 'Loth', 'Lûr', 'Mal', 'Min', 'Mor', 'Nan', 'Nar', 'Nen', 'Nor', 'Or', 'Pel', 'Quel', 'Ram', 'Rhûn', 'Riv', 'Ro', 'Sar', 'Sil', 'Sir', 'Tal', 'Tar', 'Taur', 'Thar', 'Tir', 'Um', 'Val'],
        'suffixes': ['ad', 'aeg', 'ael', 'an', 'and', 'ant', 'ar', 'ard', 'as', 'ath', 'bor', 'dan', 'del', 'dir', 'dol', 'dor', 'dras', 'duin', 'dur', 'ech', 'ed', 'eg', 'eil', 'el', 'en', 'eor', 'er', 'es', 'falas', 'fast', 'find', 'fin', 'for', 'gal', 'gard', 'garth', 'gil', 'gond', 'gor', 'had', 'har', 'hîr', 'ia', 'il', 'in', 'ion', 'ir', 'is', 'ith', 'kar', 'lad', 'laer', 'lam', 'land', 'las', 'lin', 'lith', 'lond', 'lor', 'los', 'loth', 'mar', 'men', 'mir', 'mith', 'mon', 'mor', 'nan', 'nar', 'nath', 'ndor', 'nen', 'nil', 'nir', 'nor', 'orn', 'os', 'ost', 'ram', 'randir', 'ras', 'rath', 'rhim', 'rhûn', 'rien', 'ril', 'rim', 'rin', 'rion', 'ro', 'roch', 'rond', 'ros', 'roth', 'ruin', 'sal', 'sar', 'sil', 'sir', 'thal', 'thas', 'thel', 'thir', 'thôl', 'thor', 'thuin', 'til', 'tin', 'tir', 'uial', 'uil', 'ûn', 'ur', 'val', 'van', 'waith', 'wen', 'wing', 'yale'],
        'vowels': ['a', 'e', 'i', 'o', 'u', 'ae', 'ei', 'ia', 'io', 'ua'],
        'single': ['Arnor', 'Gondor', 'Mordor', 'Eriador', 'Rohan', 'Beleriand', 'Mithlond', 'Imladris', 'Lórien', 'Isengard', 'Dol Guldur', 'Moria', 'Angmar', 'Harad', 'Rhûn', 'Dale', 'Esgaroth'],
        'types': ['Castle', 'Keep', 'Village', 'Town', 'City', 'Citadel', 'Hold', 'Fortress', 'Tower', 'Haven', 'Vale', 'Wood', 'March', 'Fief'],
        'ocean_prefixes': ['Endless', 'Great', 'Sunken', 'Whispering', 'Shrouded', 'Forgotten', 'Sundering', 'Eternal', 'Outer'],
        'ocean_suffixes': ['Expanse', 'Ocean', 'Deep', 'Void', 'Reach', 'Sea'],
        'sea_prefixes': ['Jade', 'Singing', 'Broken', 'Frozen', 'Stormy', 'Inner', 'Shining', 'Restless'],
        'mountain_prefixes': ['Dragon\'s', 'Giant\'s', 'Cloud', 'Stone', 'Iron', 'Shadow', 'Misty', 'Grey', 'Red', 'White', 'Black'],
        'mountain_suffixes': ['Spine', 'Teeth', 'Reach', 'Crown', 'Veins', 'Crags', 'Peaks', 'Hills', 'Mountains', 'Range'],
        'bay_suffixes': ['Sorrow', 'Whispers', 'Kings', 'Fools', 'Merchants', 'Drowned', 'Lost', 'Last', 'Ice']
    },
    'Sci-Fi': {
        'prefixes': ['Acheron', 'Aegis', 'Alpha', 'Andro', 'Antares', 'Apex', 'Arc', 'Astro', 'Aura', 'Azure', 'Beta', 'Bio', 'Ceres', 'Chrono', 'Cinder', 'Cognito', 'Com', 'Cryo', 'Cyber', 'Cygni', 'Delta', 'Dyna', 'Echo', 'Elysium', 'Eon', 'Epsilon', 'Exo', 'Flux', 'Forerunner', 'Galvan', 'Giga', 'Grav', 'Halcyon', 'Helios', 'Holo', 'Hyper', 'Infra', 'Inter', 'Iota', 'Juno', 'Kappa', 'Kelvin', 'Kepler', 'Kilo', 'Kodiak', 'Lambda', 'Lazarus', 'Luna', 'Mag', 'Mega', 'Meta', 'Micro', 'Myriad', 'Nano', 'Nebula', 'Neo', 'Nexus', 'Nova', 'Omni', 'Omega', 'Orion', 'Orphiuchus', 'Ortho', 'Pallas', 'Pan', 'Para', 'Penta', 'Phase', 'Pico', 'Plex', 'Poly', 'Praxis', 'Procyon', 'Proto', 'Proxy', 'Psy', 'Pylon', 'Quad', 'Quantum', 'Quasar', 'Relay', 'Rho', 'Scion', 'Sigma', 'Sol', 'Spectre', 'Star', 'Stel', 'Sub', 'Syn', 'Tau', 'Tech', 'Telsa', 'Tensor', 'Tera', 'Terra', 'Tetra', 'Thanatos', 'Thermo', 'Triton', 'Tycho', 'Ultra', 'Umbra', 'Uni', 'Vanguard', 'Vela', 'Vesta', 'Vex', 'Vita', 'Void', 'Volta', 'Vortex', 'War', 'Xeno', 'Xylo', 'Zenith', 'Zeta', 'Zion'],
        'suffixes': ['-7', ' Prime', ' Station', ' IX', ' Colony', ' Base', ' One', ' Omega', ' Terminus', ' Hub', ' Complex', ' Citadel', ' Point', ' Major', ' Minor', ' Spire', ' Array', ' Core', ' Drift', ' Facility', ' Gate', ' Grid', ' Harbor', ' Installation', ' Lab', ' Locus', ' Main', ' Node', ' Outpost', ' Port', ' Post', ' Relay', ' Ring', ' Secundus', ' Sector', ' Spoke', ' Terminal', ' Tower', ' Works', ' Zone'],
        'single': ['Aurelia', 'Elysia', 'Hadrian', 'Hyperion', 'Icarus', 'Janus', 'Kepler-186f', 'Magellan', 'Meridian', 'Olympus', 'Pandora', 'Prometheus', 'Solitude', 'Tantalus', 'Tartarus', 'Terminus', 'Vespera', 'Zion'],
        'types': ['Outpost', 'Colony', 'Station', 'Base', 'Habitat', 'Complex', 'Dome', 'Platform', 'Rig', 'Spire', 'Settlement', 'City'],
        'ocean_prefixes': ['Primary', 'Galactic', 'Azure', 'Helium', 'Forbidden', 'Xeno', 'Methane', 'Hydrogen'],
        'ocean_suffixes': ['Ocean', 'Sea', 'Expanse', 'Deep', 'Void', 'Basin'],
        'sea_prefixes': ['Cryo', 'Hydro', 'Methane', 'Irradiated', 'Plasma', 'Silicon'],
        'mountain_prefixes': ['Titanium', 'Impact', 'Cryo-volcanic', 'Tectonic', 'Obsidian', 'Iridium'],
        'mountain_suffixes': ['Range', 'Uplift', 'Plateau', 'Massif', 'Spine', 'Shield'],
        'bay_suffixes': ['Landing', 'Sector', 'Zone', 'Port', 'Anchor', 'Approach', 'Gulf']
    },
    'Post-Apocalyptic': {
        'prefixes': ['Ash', 'Barren', 'Black', 'Bleak', 'Blood', 'Bone', 'Broken', 'Burn', 'Corpse', 'Crack', 'Crimson', 'Cross', 'Dead', 'Dog', 'Dust', 'Dyer', 'Echo', 'Fade', 'Fall', 'Fort', 'Gallows', 'Ghost', 'Gloom', 'Grave', 'Grim', 'Grit', 'Hang', 'Husk', 'Iron', 'Kill', 'Last', 'Lone', 'Lost', 'Mad', 'Mire', 'Mist', 'Mud', 'Mute', 'New', 'No', 'Old', 'Pale', 'Quick', 'Quiet', 'Rag', 'Rat', 'Red', 'Rot', 'Ruin', 'Rust', 'Salt', 'Sand', 'Scab', 'Scrap', 'Shade', 'Shadow', 'Shatter', 'Shiv', 'Silent', 'Skull', 'Slick', 'Slit', 'Smoke', 'Sorrow', 'Spike', 'Still', 'Stone', 'Stray', 'Sunken', 'Tar', 'Thorn', 'Torn', 'Twist', 'Waste', 'Wreck'],
        'suffixes': ['-Town', ' Heap', ' Hope', ' Haven', ' Rock', ' Reach', ' Scrap', ' Yard', ' Camp', ' Refuge', ' Point', ' Out', ' Pit', ' Fall', ' End', ' Grave', ' Post', ' Fort', ' Scrap', 'Shanty', 'Hole', 'Den', 'Run', 'Gorge', 'Rest', 'Stop', 'Cross', 'Drift', 'Gulch', 'Sprawl', 'Yard'],
        'single': ['Salvation', 'Terminus', 'Last Hope', 'The Pitt', 'Junkyard', 'The Glow', 'Golgotha', 'Bartertown', 'Rapture', 'Bedlam', 'Elysium', 'Gomorrah', 'Bedrock'],
        'types': ['Camp', 'Settlement', 'Refuge', 'Fort', 'Scrap-town', 'Holdout', 'Warren', 'Den', 'Oasis', 'Stronghold', 'Bunker'],
        'ocean_prefixes': ['Acid', 'Sludge', 'Poison', 'Dead', 'Black', 'Blight', 'Rotten', 'Glass'],
        'ocean_suffixes': ['Sea', 'Wastes', 'Mire', 'Deep', 'Expanse', 'Blackness'],
        'sea_prefixes': ['Grime', 'Scab', 'Rust', 'Oil', 'Bile', 'Grit', 'Tar'],
        'mountain_prefixes': ['Jagged', 'Shattered', 'Glass', 'Rusted', 'Concrete', 'Bone', 'Slag', 'Barbed'],
        'mountain_suffixes': ['Peaks', 'Sprawl', 'Heap', 'Ruin', 'Ridge', 'Spine', 'Scraps'],
        'bay_suffixes': ['Graveyard', 'Wreckage', 'Grave', 'End', 'Choke', 'Grief', 'Boneyard']
    },
}