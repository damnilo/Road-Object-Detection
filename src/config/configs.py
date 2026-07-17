IMG_SIZE = 416
DOWNSAMPLE_FACTOR = 32
BOXES_PER_CELL = 2

if IMG_SIZE % DOWNSAMPLE_FACTOR:
    raise ValueError("IMG_SIZE must be divisible by DOWNSAMPLE_FACTOR")

GRID_SIZE = IMG_SIZE // DOWNSAMPLE_FACTOR
FINE_GRID_SIZE = IMG_SIZE // (DOWNSAMPLE_FACTOR // 2)
SMALL_OBJECT_AREA = 32 * 32

CLASSES = {
    0: 'car',
    1: 'bus',
    2: 'truck',
    3: 'person',
    4: 'bicycle',
    5: 'motorcycle',
    6: 'train',
    7: 'light',
    8: 'traffic sign',
    9: 'rider'
}

CLS_TO_IDX = {cls: idx for idx, cls in CLASSES}

CATEGORY_ALIASES = {
    'traffic light': 'light',
    'bike': 'bicycle',
    'motor': 'motorcycle',
    'pedestrian': 'person',
}

def normalize_category(name):
    if name is None:
        return None
    
    name = name.strip().lower()
    return CATEGORY_ALIASES.get(name, name)

NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in CLASSES.items()}

FINE_ANCHORS = [(0.03, 0.08), (0.05, 0.04)]
COARSE_ANCHORS = [(0.12, 0.10), (0.25, 0.18)]