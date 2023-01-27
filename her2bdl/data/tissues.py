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
from skimage.transform import resize
import openslide
from .wsi import (
    size_scaler,
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
    'select_roi_manual', 
    'save_segmentation',
    'save_objects', 'load_objects',
    'fit_sampling_map', 'save_sampling_map'
]



"""
Tissue Segmentation
--------------------
"""

def apply_tissues_segmentation(slide_or_image_rgb, level=None, size=None):
    """
    Compute foreground and background mask by apply threshold otsu.

    Parameters
    ==========
    slide_or_image_rgb : `openslide.OpenSlide` or `np.darray`
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
    if isinstance(slide_or_image_rgb, np.ndarray):
        size = slide_or_image_rgb.shape[:2][::-1]
        image = slide_or_image_rgb
    else:
        assert (level is None) != (size is None), "Select level or size"
        size = slide_or_image_rgb.level_dimensions[level] if level is not None else size
        image = pil_to_np_rgb(slide_or_image_rgb.get_thumbnail(size))
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


def apply_labeled_segmentation(slide_or_image, level=None, size=None):
    """
    Select ROIs from WSI using a GUI.
    Parameters
    ==========
    slide_or_image : `openslide.OpenSlide` or `np.darray`
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
    if isinstance(slide_or_image, np.ndarray):
        size = slide_or_image.shape[:2][::-1]
        image = slide_or_image
    else:
        size = slide_or_image.level_dimensions[level] if level is not None else size
        image = np.array(slide_or_image.get_thumbnail(size))
    segmentation = apply_tissues_segmentation(slide_or_image, size=size)
    label_image = label(segmentation)
    labels_to_rgb = label2rgb(label_image, image=image, bg_label=0)
    return segmentation, label_image, labels_to_rgb

def save_segmentation(case_no, labeled_segmentation, output):
    # Paths
    segmentation_folder = path.join(output, "segmentation")
    segmentation_path = path.join(segmentation_folder, f"{case_no}.npy")

    os.makedirs(segmentation_folder, exist_ok=True)
    np.save(segmentation_path, labeled_segmentation)

    return segmentation_path

def select_roi_manual(slide,  level=None, size=None, include_guess=True, title=None, display=None):
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
    display : `GUI` or `None`
        User interface api.
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
            if region.area > WSI_MIN_REGION_AREA: 
                selected = True
                is_guess = True
                ROIs_guess.append([selected, is_guess, region.label, region.centroid, region.bbox])
    new_label = len(ROIs_guess)
    ROIs_manually = []
    
    if display is not None:
        if include_guess:
            display.imshow("Label Segmentation rgb", labels_to_rgb)
            display.imshow("Segmentation", segmentation)
        title = title or "ROI Selector"
        ROIs = display.interactive_roi_selection(title, image, ROIs_manually, ROIs_guess)
        if include_guess:
            display.close_canvas(["Label Segmentation rgb", "Segmentation"])
        display.close_canvas(title)
    else:
        ROIs = ROIs_guess

    return ROIs, labeled_segmentation

def save_objects(objects, image_ihc, output):
    objects_path = path.join(output, f"objects_{image_ihc}.csv")
    objects.to_csv(objects_path, index=False)
    return objects_path

def load_objects(objects_path, filter_selected=True):
    objects = pd.read_csv(objects_path)
    if filter_selected:
        return objects[objects["selected"]]
    return objects

"""
Tissue Sampling
--------------------
"""

def fit_sampling_map(slide, object, level=None, size=None, mode='uniform', display=None):
    assert (level is None) != (size is None), "Select level or size"
    size = slide.level_dimensions[level] if level is not None else size

    # ROI
    src_box = object["min_row"], object["min_col"], object["max_row"], object["max_col"]
    
    # Open source image
    src_size, src_level = (object["segmentation_width"], object["segmentation_height"]), object["segmentation_level"]
    src_image = cv2.cvtColor(np.array(slide.get_thumbnail(src_size)),  cv2.COLOR_RGB2BGR)
    
    # Source segmentation 
    min_row, min_col, max_row, max_col = list(map(int, src_box))
    src_segmentation = np.load(object["segmentation_path"])
    src_roi_segmentation = src_segmentation[min_row:max_row, min_col:max_col]
    src_roi_segmentation[src_roi_segmentation != object["label"]] = 0
    if not object["is_guess"]: src_roi_segmentation[:,:] = 1
    src_roi_segmentation = src_roi_segmentation.astype(bool)

    # Source ROI
    src_roi = src_image[min_row:max_row, min_col:max_col]

    # Output sampling map
    dst_size, dst_level = size, level
    dst_box = size_scaler(src_box, src_size, dst_size, output_type="tuple")
    sampling_map_shape = int(max_row - min_row), int(max_col - min_col)

    # refine segmentation
    # dst_roi = np.array(slide.get_thumbnail(dst_size)) #rgb
    # min_row, min_col, max_row, max_col = dst_box
    # dst_roi = dst_roi[min_row:max_row, min_col:max_col]
    # dst_roi_segmentation = apply_tissues_segmentation(dst_roi, level=None, size=None)
    
    if mode == "uniform":
        min_row, min_col, max_row, max_col = dst_box
        dst_roi_segmentation = resize(src_roi_segmentation, ((max_row - min_row), (max_col - min_col)), order=0, preserve_range=True)
        sampling_map = dst_roi_segmentation / np.sum(dst_roi_segmentation > 0)
        
    if display is not None:
        # Include images to display
        dst_roi = cv2.cvtColor(np.array(slide.get_thumbnail(dst_size)),  cv2.COLOR_RGB2BGR) #TODO use get_region
        dst_roi = dst_roi[min_row:max_row, min_col:max_col]
        dst_roi[sampling_map == 0] = 0
        display.imshow("Src ROI", src_roi)
        display.imshow("Src Segmentation", 255*(src_roi_segmentation.astype('uint8')))
        display.imshow("Dst ROI", dst_roi)
        display.imshow("Dst Sampling map", (255*(sampling_map/sampling_map.max())).astype("uint8"))
        display.wait()
        display.close_canvas(canvas_names=["Src ROI", "Src Segmentation", "Dst ROI", "Dst Sampling map"])

    return sampling_map
    
def save_sampling_map(case_no, label, sampling_map, output):
    # Paths
    sampling_maps_folder = path.join(output, "sampling_maps")
    sampling_map_filepath = path.join(sampling_maps_folder, f"{case_no}_{label}.npy")
    
    # Save segmentatio file
    os.makedirs(sampling_maps_folder, exist_ok=True)
    np.save(sampling_map_filepath, sampling_map)

    return sampling_map_filepath