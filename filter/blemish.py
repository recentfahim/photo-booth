# Author : Rajathithan Rajasekar
# Blemish removal using sobel filter


# selected blemish patch from mouse click
def selectedBlemish(x, y, r):
    global i
    crop_img = source[y:(y + 2 * r), x:(x + 2 * r)]
    # i = i + 1
    # cv2.imwrite("blemish-"+ str(i) +".png",crop_img)
    return identifybestPatch(x, y, r)


# get the best gradient patch around the blemish
def identifybestPatch(x, y, r):
    # Nearby Patches in all 8 directions
    patches = {}

    key1tup = appendDictionary(x + 2 * r, y)
    patches['Key1'] = (x + 2 * r, y, key1tup[0], key1tup[1])

    key2tup = appendDictionary(x + 2 * r, y + r)
    patches['Key2'] = (x + 2 * r, y + r, key2tup[0], key2tup[1])

    key3tup = appendDictionary(x - 2 * r, y)
    patches['Key3'] = (x - 2 * r, y, key3tup[0], key3tup[1])

    key4tup = appendDictionary(x - 2 * r, y - r)
    patches['Key4'] = (x - 2 * r, y - r, key4tup[0], key4tup[1])

    key5tup = appendDictionary(x, y + 2 * r)
    patches['Key5'] = (x, y + 2 * r, key5tup[0], key5tup[1])

    key6tup = appendDictionary(x + r, y + 2 * r)
    patches['Key6'] = (x + r, y + 2 * r, key6tup[0], key6tup[1])

    key7tup = appendDictionary(x, y - 2 * r)
    patches['Key7'] = (x, y - 2 * r, key7tup[0], key7tup[1])

    key8tup = appendDictionary(x - r, y - 2 * r)
    patches['Key8'] = (x - r, y - 2 * r, key8tup[0], key8tup[1])

    # print(patches)
    findlowx = {}
    findlowy = {}
    for key, (x, y, gx, gy) in patches.items():
        findlowx[key] = gx

    for key, (x, y, gx, gy) in patches.items():
        findlowy[key] = gy

    y_key_min = min(findlowy.keys(), key=(lambda k: findlowy[k]))
    x_key_min = min(findlowx.keys(), key=(lambda k: findlowx[k]))

    if x_key_min == y_key_min:
        return patches[x_key_min][0], patches[x_key_min][1]
    else:
        # print("Return x & y conflict, Can take help from FFT")
        return patches[x_key_min][0], patches[x_key_min][1]


# Get the gradients of x and y
def appendDictionary(x, y):
    crop_img = source[y:(y + 2 * r), x:(x + 2 * r)]
    gradient_x, gradient_y = sobelfilter(crop_img)
    return gradient_x, gradient_y


# Apply sobel filter
def sobelfilter(crop_img):
    sobelx64f = cv2.Sobel(crop_img, cv2.CV_64F, 1, 0, ksize=3)
    abs_xsobel64f = np.absolute(sobelx64f)
    sobel_x8u = np.uint8(abs_xsobel64f)
    gradient_x = np.mean(sobel_x8u)

    sobely64f = cv2.Sobel(crop_img, cv2.CV_64F, 0, 1, ksize=3)
    abs_ysobel64f = np.absolute(sobely64f)
    sobel_y8u = np.uint8(abs_ysobel64f)
    gradient_y = np.mean(sobel_y8u)

    return gradient_x, gradient_y


# remove the blemish
def blemishRemoval(action, x, y, flags, userdata):
    # Referencing global variables
    global r, source
    # Action to be taken when left mouse button is pressed
    if action == cv2.EVENT_LBUTTONDOWN:
        # Mark the center
        blemishLocation = (x, y)
        # print(blemishLocation)
        newX, newY = selectedBlemish(x, y, r)
        newPatch = source[newY:(newY + 2 * r), newX:(newX + 2 * r)]
        cv2.imwrite("newpatch.png", newPatch)
        # Create mask for the new Patch
        mask = 255 * np.ones(newPatch.shape, newPatch.dtype)
        source = cv2.seamlessClone(newPatch, source, mask, blemishLocation, cv2.NORMAL_CLONE)
        cv2.imshow("Blemish Removal App", source)

    # Action to be taken when left mouse button is released
    elif action == cv2.EVENT_LBUTTONUP:

        cv2.imshow("Blemish Removal App", source)


if __name__ == '__main__':
    import cv2
    import numpy as np

    # Lists to store the points
    r = 15
    i = 0
    source = cv2.imread("2.jpeg", 1)
    # Make a dummy image, will be useful to clear the drawing
    dummy = source.copy()
    cv2.namedWindow("Blemish Removal App")
    # cv2.resizeWindow("Blemish Removal App",900,900)
    # highgui function called when mouse events occur
    print(f"Using a patch of radius {r}:")
    cv2.setMouseCallback("Blemish Removal App", blemishRemoval)
    k = 0
    # loop until escape character is pressed
    while k != 27:
        cv2.imshow("Blemish Removal App", source)
        k = cv2.waitKey(20) & 0xFF
        # Another way of cloning
        if k == 99:
            source = dummy.copy()
    cv2.destroyAllWindows()