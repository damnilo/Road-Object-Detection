IMG_SIZE = 416
DOWNSAMPLE_FACTOR = 32
BOXES_PER_CELL = 2

if IMG_SIZE % DOWNSAMPLE_FACTOR:
    raise ValueError("IMG_SIZE must be divisible by DOWNSAMPLE_FACTOR")

GRID_SIZE = IMG_SIZE // DOWNSAMPLE_FACTOR

CLASSES = {
    0: "car",
    1: "van",
    2: "truck",
    3: "pedestrian",
    4: "Person_sitting",
    5: "cyclist",
    6: "tram",
    7: "misc"
}

NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in CLASSES.items()}
