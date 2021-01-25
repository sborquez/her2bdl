"""
Data and dataset constants
==========================

Collections of variables for datasets and data processing.
"""

DEBUG = False
SEED = 42

# Viz
COLORS = list("rgby")

# Dataset columns roles
GROUND_TRUTH_FILE = 'groundTruth.xlsx'
IMAGE_FILES = ("{CaseNo:02d}_HER2.ndpi", "{CaseNo:02d}_HE.ndpi") 

# Input
INPUTS  = ['image_her2']
INPUT  = INPUTS[0] # if it is sigle input
# Patch
PATCH_SIZE = (256, 256)
PATCH_LEVEL = 2
PATCH_RELEVANT_RATIO = 0.8 # foreground pixels/background pixels

# Target
TARGETS = ['HeR2 SCORE']
TARGET_LABELS = {
    #value: label_name(str)
    0: '0',
    1: '1+',
    2: '2+',
    3: '3+',
}
TARGET_LABELS_list = [
    '0',
    '1+',
    '2+',
    '3+'
]
TARGET_TO_ONEHOT = {
    0: (1, 0, 0, 0),
    1: (0, 1, 0, 0),
    2: (0, 0, 1, 0),
    3: (0, 0, 0, 1)
}
TARGET = TARGETS[0] # if it is single target

# WSI
IMAGE_IHCS = ["image_her2", "image_he"]
IMAGE_IHC = "image_her2"


#WSI_DEFAULT_MAX_SIZE = (int(.8*1080), int(.8*1920))

WSI_SEGMENTATION_LEVEL = (6 if not DEBUG else 8)
WSI_SAMPLING_MAP_LEVEL = (3 if not DEBUG else 5)
WSI_SAMPLING_LEVEL = (0 if not DEBUG else 2)
WSI_MIN_REGION_AREA = (2000 if not DEBUG else 500)
