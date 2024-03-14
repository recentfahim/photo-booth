import cv2
import matplotlib.pyplot as plt
import numpy as np
import typing


class PencilSketch:
    """Apply pencil sketch effect to an image
    """
    def __init__(
        self,
        blur_simga: int = 5,
        ksize: typing.Tuple[int, int] = (0, 0),
        sharpen_value: int = None,
        kernel: np.ndarray = None,
        ) -> None:
        """
        Args:
            blur_simga: (int) - sigma ratio to apply for cv2.GaussianBlur
            ksize: (float) - ratio to apply for cv2.GaussianBlur
            sharpen_value: (int) - sharpen value to apply in predefined kernel array
            kernel: (np.ndarray) - custom kernel to apply in sharpen function
        """
        self.blur_simga = blur_simga
        self.ksize = ksize
        self.sharpen_value = sharpen_value
        self.kernel = np.array([[0, -1, 0], [-1, sharpen_value,-1], [0, -1, 0]]) if kernel == None else kernel

    def dodge(self, front: np.ndarray, back: np.ndarray) -> np.ndarray:
        """The formula comes from https://en.wikipedia.org/wiki/Blend_modes
        Args:
            front: (np.ndarray) - front image to be applied to dodge algorithm
            back: (np.ndarray) - back image to be applied to dodge algorithm

        Returns:
            image: (np.ndarray) - dodged image
        """
        result = back*255.0 / (255.0-front)
        result[result>255] = 255
        result[back==255] = 255
        return result.astype('uint8')

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """Sharpen image by defined kernel size
        Args:
            image: (np.ndarray) - image to be sharpened

        Returns:
            image: (np.ndarray) - sharpened image
        """
        if self.sharpen_value is not None and isinstance(self.sharpen_value, int):
            inverted = 255 - image
            return 255 - cv2.filter2D(src=inverted, ddepth=-1, kernel=self.kernel)

        return image

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Main function to do pencil sketch
        Args:
            frame: (np.ndarray) - frame to excecute pencil sketch on

        Returns:
            frame: (np.ndarray) - processed frame that is pencil sketch type
        """
        grayscale = np.array(np.dot(frame[..., :3], [0.299, 0.587, 0.114]), dtype=np.uint8)
        grayscale = np.stack((grayscale,) * 3, axis=-1) # convert 1 channel grayscale image to 3 channels grayscale

        inverted_img = 255 - grayscale

        blur_img = cv2.GaussianBlur(inverted_img, ksize=self.ksize, sigmaX=self.blur_simga)

        final_img = self.dodge(blur_img, grayscale)

        sharpened_image = self.sharpen(final_img)

        return sharpened_image


def img2sketch(photo, k_size):
    # Read Image
    img = cv2.imread(photo)

    # Convert to Grey Image
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert Image
    invert_img = cv2.bitwise_not(grey_img)
    # invert_img=255-grey_img

    # Blur image
    blur_img = cv2.GaussianBlur(invert_img, (k_size, k_size), 0)

    # Invert Blurred Image
    invblur_img = cv2.bitwise_not(blur_img)
    # invblur_img=255-blur_img

    # Sketch Image
    sketch_img = cv2.divide(grey_img, invblur_img, scale=256.0)

    # Save Sketch
    cv2.imwrite('sketch.png', sketch_img)

    # Display sketch
    cv2.imshow('sketch image', sketch_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# img2sketch(photo='images/2.jpg', k_size=9)


def pencil_sketch(image, *args, **kwargs):
    # open an image using opencv
    imgOriginal = cv2.imread('images/1.png')
    #img = cv2.resize(imgOriginal, (324, 720))
    img = imgOriginal
    cv2.imshow('Original', img)

    # get image height and width
    height, width, channels = img.shape

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (255, 255), 0, 0)
    img_blend = cv2.divide(img_gray, img_blur, scale=256)
    cv2.imshow('Pencil Sketch', img_blend)

    # save image using opencv
    cv2.imwrite('OfficePencilSketch.jpg', img)

    # key controller
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cartoon_filter(image, *args, **kwargs):
    img = cv2.imread(image)
    # Display the original image
    cv2.imshow("Original Image", img)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply median blur to smooth the image
    smooth_img = cv2.medianBlur(gray_img, 5)

    # Apply adaptive thresholding to detect edges
    edges = cv2.adaptiveThreshold(smooth_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 2)

    # Apply bilateral filter to create the cartoon effect
    color_img = cv2.bilateralFilter(img, 9, 300, 300)

    # Apply bitwise AND to combine the edge-detected image with the cartoon-like image
    cartoon_img = cv2.bitwise_and(color_img, color_img, mask=edges)

    # Display the cartoonized image
    cv2.imshow("Cartoonized Image", cartoon_img)

    # Save the cartoonized image
    cv2.imwrite("cartoonized.jpg", cartoon_img)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(cartoon_img, (5, 5), 0)

    # Save the output image
    cv2.imwrite('reduced_noise.jpg', blurred)


def remove_hotspots(image, *args, **kwargs):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the image to smooth out the edges
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image to create a binary image where the hot-spots are white and the rest of the image is black
    threshold = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY)[1]

    # Dilate the binary image to fill in any small holes in the hot-spots
    dilated = cv2.dilate(threshold, None, iterations=2)

    # Find the contours in the dilated image
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Set minimum size and aspect ratio for contours to be removed
    min_size = 100
    min_aspect_ratio = 0.1

    # Iterate over the contours and remove any that are not large enough or have an aspect ratio that is too small
    for i in range(len(contours)):
        contour = contours[i]
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / float(h)
        if w*h < min_size or aspect_ratio < min_aspect_ratio:
            cv2.drawContours(image, contours, i, 0, -1)

    return image


def remove_background(image):
    from rembg import remove
    from PIL import Image
    img = cv2.imread(image)
    output = remove(img)
    cv2.imwrite('bg_removed5.png', output)


# remove_background('images/5.jpg')