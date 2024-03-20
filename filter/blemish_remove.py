from rembg import remove
import cv2
from utils import filter_image_output
import numpy as np


def fix_blemish(image, source_pos, target_pos, brush_size):
    image_original = image.copy()
    # Get ROI

    clone_source_roi = image_original[source_pos[1] - brush_size:source_pos[1] + brush_size,
                       source_pos[0] - brush_size:source_pos[0] + brush_size]
    print(clone_source_roi)
    # Get mask
    clone_source_mask = np.ones(clone_source_roi.shape, clone_source_roi.dtype) * 255
    # Feather mask
    clone_source_mask = cv2.GaussianBlur(clone_source_mask, (5, 5), 0, 0)

    # Apply clone
    fix = cv2.seamlessClone(clone_source_roi, image_original, clone_source_mask, target_pos, cv2.NORMAL_CLONE)
    return fix




img = cv2.imread('1.jpg')
image = cv2.circle(img, (335, 151), 10, (0,0,0), -1)
image = fix_blemish(img, (328 - 15, 144 - 14), (328, 144), 10)
image = cv2.circle(image, (328, 400), 7, (0,0,0), -1)
image = cv2.circle(image, (163, 363), 5, (0,0,0), -1)
image = cv2.circle(image, (205, 382), 8, (0,0,0), -1)
image = cv2.circle(image, (341, 332), 7, (0,0,0), -1)
image = cv2.circle(image, (182, 396), 10, (0,0,0), -1)
image = cv2.circle(image, (302, 419), 6, (0,0,0), -1)


cv2.imshow("True Indeed", image)
cv2.waitKey(0)
