FONT_SIZE = 48
IMAGE_WIDTH = 1000
BASE_HEIGHT = 150
PADDING = 40


CONFIGS = {

    "PLAIN": {

        # --- Layout ---
        "background_path": None,

        # --- Engine geometry ---
        "curve_range": (0.0, 0.0),
        "char_spacing_range": (0, 0),
        "stroke_dilate_prob": 0.00,
        "stroke_erode_prob": 0.0,
        "line_jitter": 0,
        "height_safety_factor": 1.05,
        "width_safety_factor": 1.05,

        # --- Augmentations ---
        "perspective": False,
        "elastic": False,
        "rotation": (-1, 1),
        "dropout": False,
        "motion_blur": False,
        "noise": False,
        "jpeg": False,
        "low_res": False,
        "ink_fade": False,
    },


    "OLD_DOCUMENT": {

        # --- Layout ---
        "background_path": "background/bg1.png",

        # --- Engine geometry ---
        "curve_range": (-0.0002, 0.0001),  # was (-0.0025, 0.0025)
        "char_spacing_range": (-2, 4),
        "stroke_dilate_prob": 0.4,
        "stroke_erode_prob": 0.25,
        "line_jitter": 3,
        "height_safety_factor": 1.2,
        "width_safety_factor": 1.2,

        # --- Augmentations ---
        "perspective": True,
        "elastic": True,
        "rotation": (-2, 2),
        "dropout": True,
        "motion_blur": False,
        "noise": True,
        "jpeg": True,
        "low_res": True,
        "ink_fade": True,
    },
}