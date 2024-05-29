import os
import numpy as np
from PIL import Image as im
from typing import Dict, List, Optional, Tuple


def nice_box(width: int, contents: Optional[List[Tuple[str, Optional[str]]]] = None, margin: int = 2, padding: int = 0,
             thick: bool = False, upper: bool = True, lower: bool = True, sides: bool = True) -> str:
    """
    Creates a nice decorative text box with the following options:

    Args:
        width (int): The total width of the box (including margins).
        contents (list, optional): A list of strings to be displayed inside the box. 
            Each string can optionally have a second element specifying alignment 
            ('c' for center, 'r' for right, default is left). Defaults to None.
        margin (int, optional): The margin size around the contents. Defaults to 2.
        padding (int, optional): The padding left of the text box. Defaults to 0.
        thick (bool, optional): Boolean, determines if thick or thin box characters are used. Defaults to False.
        upper (bool, optional): Boolean, controls if the top border is drawn. Defaults to True.
        lower (bool, optional): Boolean, controls if the bottom border is drawn. Defaults to True.
        sides (bool, optional): Boolean, controls if the side borders are drawn. Defaults to True.

    Returns:
        str: The complete box string.
    """
    if not isinstance(width, int) or not isinstance(margin, int):
        raise TypeError('Width and margin must be integers.')

    if width <= 2 * margin:
        raise ValueError('Width must be greater than twice the margin.')

    contents = [] if contents is None else contents

    if not isinstance(contents, list):
        raise TypeError('Contents must be a list of tuples.')

    # Dictionary to map between thick/thin box characters
    box_char: Dict[str, str] = {'H': '═', 'V': '║', 'TL': '╔', 'TR': '╗',
                                'BL': '╚', 'BR': '╝'} if thick \
        else {'H': '─', 'V': '│', 'TL': '┌', 'TR': '┐',
              'BL': '└', 'BR': '┘'}

    # Calculate inner width for content display area
    inner_width: int = width - margin * 2

    box_str: str = ''

    if upper:
        # Add top border
        box_str += ' ' * padding + (box_char['TL'] if sides else ' ') \
            + box_char['H'] * width + (box_char['TR'] if sides else ' ')

    for content in contents:
        if not isinstance(content, tuple) or len(content) != 2 or \
                not isinstance(content[0], str) or not isinstance(content[1], str):
            raise ValueError(
                'Contents must be a list of tuples with (string, alignment) format.')
        text, alignment = content

        # Truncate content if it's longer than the inner width
        text = text[:inner_width-3] + \
            '...' if len(text) > (inner_width) else text

        match alignment:
            case 'c':
                text = text.center(inner_width)
            case 'r':
                text = text.rjust(inner_width)
            case _:
                text = text.ljust(inner_width)

        # Build the content string with margins and vertical bar
        box_str += '\n' + ' ' * padding + (box_char['V'] if sides else ' ') \
            + ' ' * margin + text + ' ' * margin + (box_char['V'] if sides else ' ')

    if lower:
        # Add bottom border
        box_str += '\n' + ' ' * padding + (box_char['BL'] if sides else ' ') \
            + box_char['H'] * width + (box_char['BR'] if sides else ' ')

    return box_str


def data_to_images(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    min_value = 0.165
    max_value = 1.0
    data_normalized = ((data - min_value) / (max_value - min_value) * 255).astype(np.uint8)
    n_digits = len(str(len(data_normalized)))

    for i, image_data in enumerate(data_normalized):
        image = im.fromarray(image_data.transpose(1, 2, 0))
        image_path = os.path.join(output_dir, f'image_{i+1:0{n_digits}d}.png')
        try:
            image.save(image_path)
        except IOError as e:
            print(f'Error saving {image_path}: {e}')


def images_to_data(input_dir):
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png') and os.path.isfile(os.path.join(input_dir, f))]
    image_files.sort()
    min_pixel_value = 0.165
    max_pixel_value = 1.0
    image_data_list = []

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        try:
            image = im.open(image_path)
            image_data = np.array(image)
            image_data_unnormalized = (image_data / 255.0) * (max_pixel_value - min_pixel_value) + min_pixel_value
            image_data_unnormalized = image_data_unnormalized.transpose(2, 0, 1)
            image_data_list.append(image_data_unnormalized)
        except IOError as e:
            print(f'Error opening or processing {image_path}: {e}')

    image_array = np.stack(image_data_list)
    return image_array
