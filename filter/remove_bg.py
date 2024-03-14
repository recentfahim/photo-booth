from rembg import remove
import cv2 as open_cv
from utils import filter_image_output


def remove_background(img_folder, img_name):
    img = open_cv.imread(img_folder + '/' + img_name)
    output = remove(img)
    open_cv.imwrite(filter_image_output(img_folder) + img_name, output)
    return output

