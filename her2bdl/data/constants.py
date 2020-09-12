"""
Data and dataset constants
==========================

Collections of variables for datasets and data processing.
"""

DEBUG=False

# Dataset columns roles
GROUND_TRUTH_FILE = 'groundTruth.xlsx'
IMAGE_FILES = ("{CaseNo:02d}_Her2.ndpi", "{CaseNo:02d}_HE.ndpi") 

# Input
INPUTS  = ['image_her2']
INPUT  = INPUTS[0] # if it is sigle input

# Target
TARGETS = ['HeR2 SCORE']
TARGET_LABELS = {
    #value: label_name(str)
    0: '0',
    1: '1+',
    2: '2+',
    3: '3+',
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
