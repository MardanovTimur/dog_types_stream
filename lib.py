import numpy as np
from keras_preprocessing.image.utils import load_img


def get_images(paths: list, WH):
    """Get resized images by <WH>"""
    return np.array([np.array(load_img(file_name, target_size=WH))
                     for file_name in paths])
