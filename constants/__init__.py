SCAN_DIM = (1, 189, 192, 192)  # C, D, H, W
SLICE_DIM = (1, 192, 192)
HEAD_MASK_THRESHOLD = 8.0
LESION_SIZES = ["No Lesion", "Small", "Medium", "Large"]
DATASET_ANCHORS = {
    "ATLAS": {
        "sizes": ((5, 32, 54, 93),),
        "aspect_ratios": ((0.3, 0.5, 1.0, 2.0),),
    },
    "ISLES-DWI": {
        "sizes": ((7, 10, 20),),
        "aspect_ratios": ((0.3, 0.5, 1.0, 2.0),),
    },
    "ISLES-FLAIR": {
        "sizes": ((5, 12, 23, 84),),
        "aspect_ratios": ((0.3, 0.5, 1.0, 2.0)),
    },
}
DATASET_PATTERNS = {
    "ATLAS": {
        "subject": r'r(\d+)',
        "session": r'r\d+s(\d+)',
    },
    "ISLES": {
        "subject": r'strokecase(\d+)',
        "session": r'ses-(\d+)'
    },
}
