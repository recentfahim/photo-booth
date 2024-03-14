import cv2 as open_cv
from utils import filter_image_output


def cantoon_filter(img_folder, img_name):
    img = open_cv.imread(img_folder + '/' + img_name)

    # Convert the image to grayscale
    gray_img = open_cv.cvtColor(img, open_cv.COLOR_BGR2GRAY)

    # Apply median blur to smooth the image
    smooth_img = open_cv.medianBlur(gray_img, 5)

    # Apply adaptive thresholding to detect edges
    edges = open_cv.adaptiveThreshold(smooth_img, 255, open_cv.ADAPTIVE_THRESH_MEAN_C, open_cv.THRESH_BINARY, 7, 2)

    # Apply bilateral filter to create the cartoon effect
    color_img = open_cv.bilateralFilter(img, 9, 300, 300)

    # Apply bitwise AND to combine the edge-detected image with the cartoon-like image
    cartoon_img = open_cv.bitwise_and(color_img, color_img, mask=edges)

    # Save the cartoonized image
    open_cv.imwrite(filter_image_output(img_folder) + img_name, cartoon_img)

    # # Apply Gaussian blur to reduce noise
    # blurred = open_cv.GaussianBlur(cartoon_img, (5, 5), 0)
    #
    # # Save the output image
    # open_cv.imwrite('reduced_noise.jpg', blurred)
    #
    #

    return cartoon_img
