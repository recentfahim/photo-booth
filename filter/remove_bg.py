from rembg import remove
import cv2 as open_cv


def remove_background(img_folder, img_name):
    img = open_cv.imread(img_folder + '/' + img_name)
    output = remove(img)
    open_cv.imwrite(img_folder + '/processed' + img_name, output)
    return output

