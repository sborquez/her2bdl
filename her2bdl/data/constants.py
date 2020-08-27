"""
Data and dataset constants
==========================

Collections of variables for datasets and data processing.
"""


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

