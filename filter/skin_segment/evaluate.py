import cv2
import torch
from torchvision import transforms
import numpy
from .fcn import FCNResNet101


def _load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_model(model_type, state_dict):
    category_prefix = '_categories.'
    categories = [k for k in state_dict.keys() if k.startswith(category_prefix)]
    categories = [k[len(category_prefix):] for k in categories]

    model = model_type(categories)
    model.load_state_dict(state_dict)

    return model


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
    image = numpy.transpose(image, (1, 2, 0))
    image = (image * img_std) + img_mean
    image = (255 * image).astype(numpy.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mask = mask.cpu().numpy()

    colours = (
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 128, 255),
        (0, 255, 128), (128, 0, 255)
    )

    for label, (category, category_mask) in enumerate(zip(categories, mask)):
        cat_image = image.copy()

        cat_colour = colours[label % len(colours)]
        cat_colour = numpy.array(cat_colour)
        cat_image[category_mask] = 0.5 * cat_image[category_mask] + 0.5 * cat_colour

        mask_image = image.copy()
        mask_image[~category_mask] = 0

        yield category, cat_image, mask_image


def evaluate(image_path):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    import os
    model = '/Users/fahim/projects/pet/photo-booth/filter/pretrained_model/model_segmentation_skin_30.pth'
    threshold = 0.7

    model = torch.load(model, map_location=device)
    model = load_model(FCNResNet101, model)
    model.to(device).eval()

    fn_image_transform = transforms.Compose(
        [
            transforms.Lambda(lambda image_path: _load_image(image_path)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    image_file = image_path
    image = fn_image_transform(image_file)

    with torch.no_grad():
        image = image.to(device).unsqueeze(0)
        results = model(image)['out']
        results = torch.sigmoid(results)

        results = results > threshold

    for category, category_image, mask_image in draw_results(image[0], results[0], categories=model.categories):
        return mask_image, category_image
