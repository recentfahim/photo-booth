import cv2
import numpy as np
from utils import filter_image_output


def whitening_method1(img_folder, img_name):
    image = cv2.imread(img_folder + '/' + img_name)

    # Convert the image to YUV color space
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Increase brightness by scaling the Y channel
    brightness_scale = 1.07
    image_yuv[:, :, 0] = np.clip(image_yuv[:, :, 0] * brightness_scale, 0, 255)

    # Convert the image back to the BGR color space
    image_output = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    # kernel = np.array([[-1, -1, -1],
    #                    [-1, 9, -1],
    #                    [-1, -1, -1]])

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    image_sharp = cv2.filter2D(src=image_output, ddepth=-1, kernel=kernel)

    cv2.imwrite(filter_image_output(img_folder) + img_name, image_sharp)

    return image_output


def whitening_method2(img_folder, img_name):
    image = cv2.imread(img_folder + '/' + img_name)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for skin color in HSV
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask for skin regions
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Apply the mask to the original image
    skin_only = cv2.bitwise_and(image, image, mask=skin_mask)

    # Convert the skin_only image to the LAB color space
    lab_skin_only = cv2.cvtColor(skin_only, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab_skin_only)

    # Increase brightness in the L channel (lightness)
    brightness_scale = 1.07
    l_channel = cv2.convertScaleAbs(l_channel, alpha=brightness_scale, beta=0)

    # Merge the channels back together
    lab_skin_brightened = cv2.merge([l_channel, a_channel, b_channel])

    # Convert the LAB image back to the BGR color space
    skin_brightened = cv2.cvtColor(lab_skin_brightened, cv2.COLOR_LAB2BGR)

    # Combine the skin_brightened image with the original non-skin regions
    non_skin_mask = cv2.bitwise_not(skin_mask)
    result_image = cv2.bitwise_or(cv2.bitwise_and(image, image, mask=non_skin_mask), skin_brightened)

    cv2.imwrite(filter_image_output(img_folder) + img_name, result_image)

    return result_image


def whitening_method3(img_folder, img_name):
    image = cv2.imread(img_folder + '/' + img_name)
    contrast = 1.05  # Contrast control ( 0 to 127)
    brightness = 2  # Brightness control (0-100)

    result_image = cv2.addWeighted(image, contrast, image, 0, brightness)

    cv2.imwrite(filter_image_output(img_folder) + img_name, result_image)

    return result_image


def whitening_method5(img_folder, img_name):
    image = cv2.imread(img_folder + '/' + img_name)

    import torch
    from torchvision import transforms
    from skin_segment.models import load_model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = torch.load('pretrained_model/model_segmentation_skin_30.pth', map_location=device)
    model = load_model(models['FCNResNet101'], model)
    model.to(device).eval()

    fn_image_transform = transforms.Compose(
        [
            transforms.Lambda(lambda image_path: _load_image(image_path)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    image = fn_image_transform(image_file)

    with torch.no_grad():
        image = image.to(device).unsqueeze(0)
        results = model(image)['out']
        results = torch.sigmoid(results)

        results = results > args.threshold

    for category, category_image, mask_image in draw_results(image[0], results[0], categories=model.categories):
        if args.save:
            output_name = f'results_{category}_{image_file.name}'
            logging.info(f'writing output to {output_name}')
            cv2.imwrite(str(output_name), category_image)
            cv2.imwrite(f'mask_{category}_{image_file.name}', mask_image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    face = face_classifier.detectMultiScale(
        image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )
    # image_output = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

    for (x, y, w, h) in face:
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
        face_roi = image[y:y + h, x:x + w]

        # Increase brightness by scaling the face ROI
        brightness_scale = 1.2
        face_roi[:, :, 0] = cv2.convertScaleAbs(face_roi[:, :, 0], alpha=brightness_scale, beta=0)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return img_name


def whitening_method4(img_folder, img_name):
    image = cv2.imread(img_folder + '/' + img_name)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    face = face_classifier.detectMultiScale(
        image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )
    # image_output = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

    for (x, y, w, h) in face:
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
        face_roi = image[y:y + h, x:x + w]

        # Increase brightness by scaling the face ROI
        brightness_scale = 1.2
        face_roi[:, :, 0] = cv2.convertScaleAbs(face_roi[:, :, 0], alpha=brightness_scale, beta=0)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return img_name


# #
# # # Convert the image to the HSV color space
# # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# #
# # # Define lower and upper bounds for skin color in HSV
# # lower_skin = np.array([0, 48, 80], dtype=np.uint8)
# # upper_skin = np.array([20, 255, 255], dtype=np.uint8)
# #
# # # Create a mask for skin regions
# # skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
# #
# # # Apply the mask to the original image
# # skin_only = cv2.bitwise_and(image, image, mask=skin_mask)
# #
# # # Convert the skin_only image to the LAB color space
# # lab_skin_only = cv2.cvtColor(skin_only, cv2.COLOR_BGR2LAB)
# #
# # # Split the LAB image into L, A, and B channels
# # l_channel, a_channel, b_channel = cv2.split(lab_skin_only)
# #
# # # Increase brightness in the L channel (lightness)
# # brightness_scale = 1.2
# # l_channel = cv2.convertScaleAbs(l_channel, alpha=brightness_scale, beta=0)
# #
# # # Merge the channels back together
# # lab_skin_brightened = cv2.merge([l_channel, a_channel, b_channel])
# #
# # # Convert the LAB image back to the BGR color space
# # skin_brightened = cv2.cvtColor(lab_skin_brightened, cv2.COLOR_LAB2BGR)
# #
# # # Combine the skin_brightened image with the original non-skin regions
# # non_skin_mask = cv2.bitwise_not(skin_mask)
# # result_image = cv2.bitwise_or(cv2.bitwise_and(image, image, mask=non_skin_mask), skin_brightened)
#
#
# cv2.imshow('Whitened Face', result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
import torch
from torchvision import transforms

def draw_results(
    image: torch.Tensor,
    mask: torch.Tensor,
    categories,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225)
):
    assert mask.shape[0] == len(categories)
    assert image.shape[1:] == mask.shape[1:]
    assert mask.dtype == torch.bool

    image = image.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image * img_std) + img_mean
    image = (255 * image).astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mask = mask.cpu().numpy()

    colours = (
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 128, 255),
        (0, 255, 128), (128, 0, 255)
    )

    for label, (category, category_mask) in enumerate(zip(categories, mask)):
        cat_image = image.copy()

        cat_colour = colours[label % len(colours)]
        cat_colour = np.array(cat_colour)
        cat_image[category_mask] = 0.5 * cat_image[category_mask] + 0.5 * cat_colour

        mask_image = image.copy()
        mask_image[~category_mask] = 0

        yield category, cat_image, mask_image