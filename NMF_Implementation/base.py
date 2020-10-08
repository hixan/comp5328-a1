from pathlib import Path
import warnings
import numpy as np
import cv2
import re
from typing import Tuple
from typing import List

class Implementation:

    def fit(self, X):
        """Train this model to represent X.

        (learn the representation dictionary in the case of NMF)

        :param X: input data (first dimension represents observations)
        :return: None
        """
        raise NotImplementedError(f'the train method on {self.__class__} '
                'has not been implemented')

    def transform(self, X):
        """transform X to its representation vectors, R. (along 1st axis)

        Raises an error if called before fit.
        """
        raise NotImplementedError(f'the fit method on {self.__class__} '
        'has not been implemented')

    def inverse_transform(self, R):
        """transform R back into its original representation space.

        Raises an error if called before fit.
        """
        raise NotImplementedError(f'the fit method on {self.__class__} ',
        'has not been implemented')

    def get_metavalues(self):
        """Returns meta information about this model including hyperparameters
        and training losses"""
        raise NotImplementedError


def load_data(root: Path, xy: Tuple[int, int] = None):
    """
    Load ORL (or Extended YaleB) dataset to numpy array.

    :param root: root directory of dataset (data/ORL or data/CroppedYaleB)
    :param xy: requested width and height of the images.
    """

    # sanitize input
    root = Path(root)
    assert root.is_dir(), f'the directory {root} does not exist.'
    '(it may exist but is not a directory). Cannot load dataset.'

    if xy:
        assert len(xy) == 2, f'xy must be None or 2 length tuple of integers. Got {xy}.'
        xy = tuple(map(int, xy))

    images, labels = [], []

    for person in root.glob('*'):
        m = re.search(r'\d+', person.name)
        if m is None or not person.is_dir():
            warnings.warn(f'could not handle person {person}. Skipping...')
            continue

        person_no = int(m.group(0))
        for image_path in person.glob('*'):
            # Remove background images in Extended YaleB dataset.
            if image_path.name.endswith('Ambient.pgm'):
                continue

            if not image_path.name.endswith('.pgm'):
                continue

            # load image.
            img = cv2.imread(str(image_path), -1)

            # resize to requested size
            if xy:
                img = cv2.resize(img, xy)

            # convert image to numpy array.

            # collect data and label.
            #images.append(img.reshape((-1,1)))
            images.append(img)
            labels.append(person_no)

    # concate all images and labels.
    #images = np.concatenate(images, axis=1)
    images = np.stack(images)
    labels = np.array(labels)

    return images, labels
