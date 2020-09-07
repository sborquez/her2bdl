"""
Whole Slide Images
==================

Collections of function to load and handler WSI.

Resources:
- https://openslide.org/api/python
- https://developer.ibm.com/articles/an-automatic-method-to-identify-tissues-from-big-whole-slide-images-pt1/
- https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_label.html#sphx-glr-auto-examples-segmentation-plot-label-py
- 
"""
import logging
import pandas as pd
import numpy as np
from os import path
import os
import shutil
import cv2
import skimage
from skimage.color import rgb2hed, label2rgb
from skimage.filters.rank import entropy
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import opening, closing
import openslide
from .wsi import (
    pil_to_np_rgb,
    filter_rgb_to_hsv, filter_hsv_to_h,
    filter_rgb_to_hed, filter_hed_to_hematoxylin, filter_hed_to_dab,
    filter_rgb_to_grayscale, filter_gray_to_entropy,
    filter_binary_closing, filter_binary_opening,
    filter_otsu_threshold, 
    filter_blue_pen, filter_green_pen, filter_red_pen, filter_black_pen
)

from .constants import *

__all__ = [
    'select_roi_manual', 'save_objects' 
]



"""
Tissue Segmentation
--------------------
"""

def apply_tissues_segmentation(slide, level=None, size=None):
    """
    Compute foreground and background mask by apply threshold otsu.

    """
    assert (level is None) != (size is None), "Select level or size"
    size = slide.level_dimensions[level] if level is not None else size
    image = pil_to_np_rgb(slide.get_thumbnail(size))
    segmentation_stack = []

    # Greed threshold
    _clahe = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(25, 25))
    green = _clahe.apply(image[:,:,1])
    segmentation_stack.append(255 - filter_otsu_threshold(green, output_type="uint8"))

    # Hue threshold
    hue = filter_hsv_to_h(filter_rgb_to_hsv(image), output_type="int")
    segmentation_stack.append(filter_otsu_threshold(hue))

    # Hermatoxylin & DAB thresholds
    hed = filter_rgb_to_hed(image)
    hermatoxylin = filter_hed_to_hematoxylin(hed)
    dab = filter_hed_to_dab(hed)
    segmentation_stack.append(filter_otsu_threshold(hermatoxylin))
    segmentation_stack.append(filter_otsu_threshold(dab))
    
    # Entropy
    gray = filter_rgb_to_grayscale(image, output_type="uint8")
    entropy = filter_gray_to_entropy(gray, output_type="uint8")
    segmentation_stack.append(filter_otsu_threshold(entropy))

    # Aggregate
    stacked = np.sum(segmentation_stack, axis=0)
    mean =  (stacked/len(segmentation_stack)).astype("uint8")
    segmentation = filter_otsu_threshold(mean)

    # Post processing
    segmentation = filter_binary_closing(segmentation, disk_size=5, iterations=1, output_type="uint8")
    segmentation = filter_binary_opening(segmentation, disk_size=5, iterations=1, output_type="uint8")

    # Marks segmentation
    #mark_segmentation = filter_blue_pen(image, output_type="bool") | filter_red_pen(image, output_type="bool")\
    #                    | filter_green_pen(image, output_type="bool") | filter_black_pen(image, output_type="bool")
    #mark_segmentation = filter_binary_closing(mark_segmentation.astype("uint8"), disk_size=5, iterations=1, output_type="bool")
    #mark_segmentation = filter_binary_opening(mark_segmentation, disk_size=5, iterations=1, output_type="bool")          
    #segmentation[~mark_segmentation] = 0

    return segmentation


def apply_labeled_segmentation(slide, level=None, size=None):
    """
    Select ROIs from WSI using a GUI.
    Parameters
    ==========
    slide : `openslide.OpenSlide`
        Opened WSI.
    size : tuple(`int`, `int`)
        Thumbnail max size.
    level : `int`
        Thumbnail level.
    Returns
    =======
    `tuple` of `np.ndarray(uint8)`
        segmentions
    """
    assert (level is None) != (size is None), "Select level or size"
    size = slide.level_dimensions[level] if level is not None else size
    segmentation = apply_tissues_segmentation(slide, size=size)
    label_image = label(segmentation)
    image = np.array(slide.get_thumbnail(size))
    labels_to_rgb = label2rgb(label_image, image=image, bg_label=0)
    return segmentation, label_image, labels_to_rgb

def get_selector_callback(guess, manually):
    def mouse_click_callback(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            for i, (selected, _, _, centroid, _) in enumerate(guess):
                if np.linalg.norm(np.array([y, x]) - np.array(centroid)) > 15: continue
                guess[i][0] = not selected
            for i, (selected, _, _, centroid, _) in enumerate(manually):
                if np.linalg.norm(np.array([y, x]) - np.array(centroid)) > 15: continue
                manually[i][0] = not selected
    return mouse_click_callback
                    

def select_roi_manual(slide,  level=None, size=None, include_guess=True):
    """
    Select ROIs from WSI using a GUI.
    Parameters
    ==========
    slide : `openslide.OpenSlide`
        Opened WSI.
    size : tuple(`int`, `int`)
        Thumbnail max size.
    level : `int`
        Thumbnail level.
    include_guess : `bool`
        Include ROIs guesses by thresholding
    Returns
    =======
    `list`, `np.ndarray(int)` 
        ROIs selected and labeled segmentation at level or size given .
    """
    assert (level is None) != (size is None), "Select level or size"
    size = slide.level_dimensions[level] if level is not None else size
    image = cv2.cvtColor(np.array(slide.get_thumbnail(size)),  cv2.COLOR_RGB2BGR)

    # Apply segmentantion to detect relevant tissues.
    ROIs_guess = []
    if include_guess:
        segmentation, labeled_segmentation, labels_to_rgb = apply_labeled_segmentation(slide, level=level)
        for region in regionprops(labeled_segmentation):
            # take regions with large enough areas
            if region.area > WSI_MIN_REGION_AREA: ROIs_guess.append([True, True, region.label, region.centroid, region.bbox])
        cv2.imshow("Segmentation", segmentation)
        cv2.imshow("Label Segmentation rgb", labels_to_rgb)
    new_label = len(ROIs_guess)
    ROIs_manually = []
    capturing = True
    while capturing:
        canvas = image.copy()
        # Select from guesses
        for selected, is_guess, label, centroid, guess_box in ROIs_guess:
            min_row, min_col, max_row, max_col =  guess_box
            if selected:
                cv2.rectangle(canvas, (min_col, min_row), (max_col, max_row), (80, 200, 0), 3)
                cv2.circle(canvas, tuple(map(int, centroid))[::-1], 6, (180, 80, 50), 12)
            else:
                cv2.rectangle(canvas, (min_col, min_row), (max_col, max_row), (90, 90, 90), 3)
                cv2.circle(canvas, tuple(map(int, centroid))[::-1], 12,  (180, 80, 50), 6)
        # Select mannualy
        for selected, is_guess, label, centroid, box in ROIs_manually:
            min_row, min_col, max_row, max_col =  box
            if selected:
                cv2.rectangle(canvas, (min_col, min_row), (max_col, max_row), (80, 200, 0), 3)
                cv2.circle(canvas, tuple(map(int, centroid))[::-1], 6, (180, 80, 50), 12)
            else:
                cv2.rectangle(canvas, (min_col, min_row), (max_col, max_row), (90, 90, 90), 3)
                cv2.circle(canvas, tuple(map(int, centroid))[::-1], 12,  (180, 80, 50), 6)
        # Show results
        windowName="Press 'space' to CONTINUE - Press'a' to add new ROI"
        cv2.imshow(windowName, canvas)
        cv2.setMouseCallback(windowName, get_selector_callback(ROIs_guess, ROIs_manually))

        key = cv2.waitKey(2)
        # Manually select
        if key == ord('a'):
            roi = cv2.selectROI(windowName, canvas, showCrosshair=True, fromCenter=False)
            box = roi[1], roi[0],  roi[1]+roi[3], roi[0]+roi[2]
            centroid = (box[0] + box[2])/2., (box[1] + box[3])/2.
            ROIs_manually.append([True, False, new_label, centroid, box])
            new_label += 1
        elif key == 32: # space
            cv2.destroyAllWindows()
            capturing = False
    ROIs = ROIs_guess + ROIs_manually
    return ROIs, labeled_segmentation

def save_objects(slide, case_no, image_ihc, objects, labeled_segmentation, output, level=None, size=None, mode="a"):
    assert (level is None) != (size is None), "Select level or size"
    size = slide.level_dimensions[level] if level is not None else size
    # Paths
    objects_filepath = path.join(output, f"objects_{image_ihc}.csv")
    segmentation_folder = path.join(output, "segmentation")
    segmentation_filepath = path.join(segmentation_folder, f"{case_no}.npy")
    
    # Save segmentatio file
    if mode == "w": 
        shutil.rmtree(segmentation_folder)
    os.makedirs(segmentation_folder, exist_ok=True)
    np.save(segmentation_filepath, labeled_segmentation)

    # Save Objects file
    if (not path.exists(objects_filepath)) or (mode == "w"):
        with open(objects_filepath, "w") as file:
            file.write(f"CaseNo,level,width,height,selected,is_guess,label,centroid_row,centroid_col,min_row,min_col,max_row,max_col\n")
    with open(objects_filepath, "a") as file:
        for obj in objects:
            selected, is_guess, label, centroid, box = obj
            centroid_row,centroid_col = centroid
            min_row, min_col, max_row, max_col =  box
            file.write(f"{case_no},{level},{size[0]},{size[1]},{selected},{is_guess},{label},{centroid_row},{centroid_col},{min_row},{min_col},{max_row},{max_col}\n")
    
def load_objects(objects_filepath, filter_selected=True):
    objects = pd.read_csv(objects_filepath)
    if filter_selected:
        return objects[objects["selected"]]
    return objects