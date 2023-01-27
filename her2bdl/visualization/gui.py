"""
GUI - Graphic User Interface
============================

Class Abstract GUI offers simple methods for 
manages user inputs and canvas.
"""

import abc


__all__ = [
    'GUI',
    'GUIcmd'
]


class GUI(metaclass=abc.ABCMeta):
    
    canvas = {}

    @abc.abstractmethod
    def wait(self, timeout=None):
        pass

    # @abc.abstractmethod
    # def interactive_wsi(self, slide, canvas_name=None):
    #     pass

    @abc.abstractmethod
    def interactive_roi_selection(self, canvas_name, image, ROIs_manually=[], ROIs_guess=[]):
        pass

    @abc.abstractmethod
    def imshow(self, canvas_nane, image, resize=None, auto_scale=True):
        pass

    # @abc.abstractmethod
    # def display_figure(self, image, canvas_nane=None, wait=False):
    #     pass
    
    @abc.abstractmethod
    def progress_bar(self, iterable, canvas_name=None, total=None):
        pass

    @abc.abstractmethod
    def close_canvas(self, canvas_names=None, close_all=False):
        pass


from datetime import datetime
from tqdm import tqdm
import numpy as np
import cv2
import skimage
from skimage.transform import rescale, resize


class GUIcmd(GUI):

    WINDOWS_MAX_SHAPE = (0.6*1080, 0.6*1920)

    def wait(self, timeout=None):
        start = datetime.now()
        waiting = True
        while waiting:  
            key = cv2.waitKey(2)
            if key == 32: # space
                waiting = False
            elif (timeout is not None) and ((datetime.now() - start).seconds > timeout):
                waiting = False

    def __get_selector_callback(self, guess, manually, scale=1):
        def mouse_click_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                for i, (selected, _, _, centroid, _) in enumerate(guess):
                    if np.linalg.norm(np.array([y, x]) - scale*np.array(centroid)) > 15: continue
                    guess[i][0] = not selected
                for i, (selected, _, _, centroid, _) in enumerate(manually):
                    if np.linalg.norm(np.array([y, x]) - scale*np.array(centroid)) > 15: continue
                    manually[i][0] = not selected
        return mouse_click_callback
        
        
    def interactive_roi_selection(self, canvas_name, image, ROIs_manually=[], ROIs_guess=[]):
        capturing = True
        new_label = len(ROIs_guess)
        
        # Resize to diplayable resolution
        i_bigger_size = np.argmax(image.shape[:2])
        i_smaller_size = 1 - i_bigger_size
        if image.shape[i_bigger_size] > GUIcmd.WINDOWS_MAX_SHAPE[i_bigger_size]:
            scale =  GUIcmd.WINDOWS_MAX_SHAPE[i_bigger_size] / image.shape[i_bigger_size]
            image_ = cv2.resize(image, (0,0), fx=scale, fy=scale) 
        elif image.shape[i_smaller_size] > GUIcmd.WINDOWS_MAX_SHAPE[i_smaller_size]:
            scale = GUIcmd.WINDOWS_MAX_SHAPE[i_smaller_size] / image.shape[i_smaller_size]
            image_ = cv2.resize(image, (0,0), fx=scale, fy=scale) 
        else:
            scale = 1.0
            image_ = image

        while capturing:
            canvas = image_.copy()
            # Select from guesses
            for selected, is_guess, label, centroid, guess_box in ROIs_guess:
                guess_box = list(map(lambda b: int(b*scale), guess_box)) #scale down\
                centroid = tuple(map(lambda c: int(c*scale), centroid))
                min_row, min_col, max_row, max_col = guess_box
                if selected:
                    cv2.rectangle(canvas, (min_col, min_row), (max_col, max_row), (80, 200, 0), 3)
                    cv2.circle(canvas, tuple(map(int, centroid))[::-1], int(6*scale), (180, 80, 50), int(12/scale))
                else:
                    cv2.rectangle(canvas, (min_col, min_row), (max_col, max_row), (90, 90, 90), 3)
                    cv2.circle(canvas, tuple(map(int, centroid))[::-1], int(12*scale),  (180, 80, 50), int(6/scale))
            # Select mannualy
            for selected, is_guess, label, centroid, box in ROIs_manually:
                box = list(map(lambda b: int(b*scale), box)) #scale down
                centroid = tuple(map(lambda c: int(c*scale), centroid))
                min_row, min_col, max_row, max_col =  box
                if selected:
                    cv2.rectangle(canvas, (min_col, min_row), (max_col, max_row), (80, 200, 0), 3)
                    cv2.circle(canvas, tuple(map(int, centroid))[::-1], int(6*scale), (180, 80, 50), int(12/scale))
                else:
                    cv2.rectangle(canvas, (min_col, min_row), (max_col, max_row), (90, 90, 90), 3)
                    cv2.circle(canvas, tuple(map(int, centroid))[::-1], int(12*scale),  (180, 80, 50), int(6/scale))
            # Show results
            windowName = f"{canvas_name} - Press 'space' to CONTINUE - Press 'a' to add new ROI"
            self.canvas[canvas_name] = windowName
            cv2.imshow(windowName, canvas)
            cv2.setMouseCallback(windowName, self.__get_selector_callback(ROIs_guess, ROIs_manually, scale=scale))

            key = cv2.waitKey(2)
            # Manually select
            if key == ord('a'):
                roi = cv2.selectROI(windowName, canvas, showCrosshair=True, fromCenter=False)
                roi = list(map(lambda r: r/scale, roi))  # scale up
                box = roi[1], roi[0],  roi[1]+roi[3], roi[0]+roi[2]
                centroid = (box[0] + box[2])/2., (box[1] + box[3])/2.
                ROIs_manually.append([True, False, new_label, centroid, box])
                new_label += 1
            elif key == 32: # space
                capturing = False

        ROIs = ROIs_guess + ROIs_manually
        return ROIs

    def progress_bar(self, iterable, total=None):
        total = total if total is not None else len(iterable)
        for values in tqdm(iterable, total=total):
            yield values

    def imshow(self, canvas_nane, image, resize=None, auto_scale=True):
        windowsName = canvas_nane
        self.canvas[canvas_nane] = windowsName
        if resize:
            cv2.imshow(
                windowsName,
                cv2.resize(image, resize)
            )
        elif auto_scale:
            i_bigger_size = np.argmax(image.shape[:2])
            i_smaller_size = 1 - i_bigger_size
            if image.shape[i_bigger_size] > GUIcmd.WINDOWS_MAX_SHAPE[i_bigger_size]:
                scale =  GUIcmd.WINDOWS_MAX_SHAPE[i_bigger_size] / image.shape[i_bigger_size]
                image_ = cv2.resize(image, (0,0), fx=scale, fy=scale) 
            elif image.shape[i_smaller_size] > GUIcmd.WINDOWS_MAX_SHAPE[i_smaller_size]:
                scale = GUIcmd.WINDOWS_MAX_SHAPE[i_smaller_size] / image.shape[i_smaller_size]
                image_ = cv2.resize(image, (0,0), fx=scale, fy=scale) 
            else:
                image_ = image
            cv2.imshow(
                windowsName,
                image_
            )
        else:
            cv2.imshow(
                windowsName,
                image
            )


    def close_canvas(self, canvas_names=None, close_all=False):
        if close_all:
            cv2.destroyAllWindows()
            self.canvas = {}
        elif isinstance(canvas_names, str):
            canvas_names = [canvas_names]
        elif isinstance(canvas_names, list):
            pass
        else:
            raise ValueError("canvas_name is `None`")

        for canvas_name in canvas_names:
            if canvas_name not in self.canvas: continue
            windowsName = self.canvas[canvas_name]
            cv2.destroyWindow(windowsName)
            del self.canvas[canvas_name]

                    


