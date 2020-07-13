import random
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


__all__ = ['add_random_lines',
           'add_random_padding',
           'crop_image',
           'full_augmentation',
           'modify_brightness',
           'modify_contrast',
           'modify_sharpness',
           'random_resize',
           'random_rotation']


def full_augmentation(img: Image.Image, max_width: int) -> Image.Image:
    """
    A wrapper combining augmentation functions

    Args:
        img: Image to modify
        max_width: Maximum allowed width - used during resizing

    Returns:
        A modified Pillow Image.
    """
    img = random_resize(img, max_width=max_width)
    img = modify_sharpness(img)
    img = modify_contrast(img)
    img = modify_brightness(img)
    img = add_random_padding(img)
    img = crop_image(img)
    img = add_random_lines(img)
    img = random_rotation(img)

    return img


def add_random_lines(img: Image.Image, num_lines_range: Tuple[int, int] = (1, 3)) -> Image.Image:
    """
    Add random lines mimicking scanning artifacts.

    Args:
        img: Image to modify
        num_lines_range: Minimum and maximum number of lines to be added

    Returns:
        A modified Pillow Image.
    """
    img = np.array(img)
    H, W = img.shape[:2]

    num_lines = random.randint(*num_lines_range)
    probability = random.random()

    # Only 50% of the images will have lines added
    if probability > 0.5:

        for _ in range(num_lines):
            xmin = random.randint(1, W)
            ymin = random.randint(1, H)
            xmax = random.randint(xmin, W)
            ymax = random.randint(ymin, H)

            probability = random.random()
            thickness = 1

            if probability > 0.95:
                # 5% of the time make the line white
                color = [255] * 3
            elif probability > 0.9:
                # 5% of the time make the line black
                color = [0] * 3
            elif probability > 0.7:
                # 20% of the time make the line gray of random intensity
                color = [random.randint(1, 255)] * 3
            else:
                # 70% of the time make the line match the most common color in the snippet (typically background)
                thickness = random.randint(1, 2)
                color = int(np.bincount(img.flatten()).argmax())

            cv2.line(img, (xmin, ymin), (xmax, ymax), color, thickness=thickness)

    return Image.fromarray(img)

def add_random_padding(img: Image.Image,
                       pad_top_range: Tuple[int, int] = (0, 10),
                       pad_right_range: Tuple[int, int] = (0, 10),
                       pad_bottom_range: Tuple[int, int] = (0, 10),
                       pad_left_range: Tuple[int, int] = (0, 10)) -> Image.Image:
    """
    Add random amount of padding to the image.

    Args:
        img: Image to modify
        pad_top_range: (min,max) range specifying the allowed percentage of image height to be added as padding (int)
        pad_right_range: (min,max) range specifying the allowed percentage of image width to be added as padding (int)
        pad_bottom_range: (min,max) range specifying the allowed percentage of image height to be added as padding (int)
        pad_left_range: (min,max) range specifying the allowed percentage of image width to be added as padding (int)

    Returns:
        A modified Pillow Image.
    """
    delta_top = random.randint(*pad_top_range)
    delta_right = random.randint(*pad_right_range)
    delta_bottom = random.randint(*pad_bottom_range)
    delta_left = random.randint(*pad_left_range)

    padding = (delta_top, delta_right, delta_bottom, delta_left)

    # Find the most common color in the image
    color = int(np.bincount(np.array(img).flatten()).argmax())

    return ImageOps.expand(img, padding, fill=color)

def crop_image(img: Image.Image,
               top_range: Tuple[int, int] = (0, 5),
               right_range: Tuple[int, int] = (95, 100),
               bottom_range: Tuple[int, int] = (95, 100),
               left_range: Tuple[int, int] = (0, 5)) -> Image.Image:
    """
    Random cropping of the image

    Args:
        img: Image to modify
        top_range: (min,max) range specifying the beginning of image, expressed as percentage of image height (int)
        right_range: (min,max) range specifying the end of image, expressed as percentage of image width (int)
        bottom_range: (min,max) range specifying the end of image, expressed as percentage of image height (int)
        left_range: (min,max) range specifying the beginning of image, expressed as percentage of image width (int)

    Returns:
        A modified Pillow Image.
    """
    top = int(np.round(random.randint(*top_range) / 100 * img.size[1]))
    left = int(np.round(random.randint(*left_range) / 100 * img.size[0]))
    right = int(np.round(random.randint(*right_range) / 100 * img.size[0]))
    bottom = int(np.round(random.randint(*bottom_range) / 100 * img.size[1]))

    image = img.crop(box=(left, top, right, bottom))

    return image

def modify_brightness(img: Image.Image, factor_range: Tuple[int, int] = (0.75, 1.25)) -> Image.Image:
    """

    Args:
        img: Image to modify
        factor_range: (min,max) range specifying the allowed value of the factor.

    Returns:
        A modified Pillow Image.
    """
    factor = np.random.uniform(*factor_range)

    return ImageEnhance.Brightness(img).enhance(factor)

def modify_contrast(img: Image.Image, factor_range: Tuple[int, int] = (0.75, 1.25)) -> Image.Image:
    """

    Args:
        img: Image to modify
        factor_range: (min,max) range specifying the allowed value of the factor.

    Returns:
        A modified Pillow Image.
    """
    factor = np.random.uniform(*factor_range)

    return ImageEnhance.Contrast(img).enhance(factor)

def random_resize(img: Image.Image, max_width: int, factor_range: Tuple[int, int] = (0.9, 1.1)) -> Image.Image:
    """

    Args:
        img: Image to modify
        max_width: Maximum allowed width
        factor_range: (min,max) range specifying the allowed value of the factor.

    Returns:
        A modified Pillow Image.
    """
    factor = np.random.uniform(*factor_range)
    (width, height) = (int(img.width * factor), int(img.height * factor))

    # make sure not to exceed the max_width
    if width > max_width:
        height = int(max_width / width * height)
        width = max_width

    return img.resize((width, height))

def random_rotation(img: Image.Image, angle_range: Tuple[float, float] = (-1,1)) -> Image.Image:
    """

    Args:
        img: Image to modify
        angle_range: (min,max) range specifying allowed rotation angle.

    Returns:
        A modified Pillow Image.
    """
    angle = np.random.uniform(*angle_range)

    return img.rotate(angle)

def modify_sharpness(img: Image.Image, factor_range: Tuple[int, int] = (0.75, 1.25)) -> Image.Image:
    """

     Args:
         img: Image to modify
         factor_range: (min,max) range specifying the allowed value of the factor.

     Returns:
         A modified Pillow Image.
    """
    factor = np.random.uniform(*factor_range)

    return ImageEnhance.Sharpness(img).enhance(factor)
