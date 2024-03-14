import cv2 as open_cv
from utils import filter_image_output


def pencil_sketch(img_folder, img_name):
    imgOriginal = open_cv.imread(img_folder + '/' + img_name)
    # img = cv2.resize(imgOriginal, (324, 720))
    # img = imgOriginal
    # open_cv.imshow('Original', imgOriginal)

    # get image height and width
    height, width, channels = imgOriginal.shape

    img_gray = open_cv.cvtColor(imgOriginal, open_cv.COLOR_RGB2GRAY)
    img_blur = open_cv.GaussianBlur(img_gray, (255, 255), 0, 0)
    img_blend = open_cv.divide(img_gray, img_blur, scale=256)

    open_cv.imwrite(filter_image_output(img_folder) + img_name, img_blend)

    return img_blend
