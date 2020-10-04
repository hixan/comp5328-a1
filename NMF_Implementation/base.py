from pathlib import Path
import warnings
import numpy as np
import cv2
import re
from typing import Tuple

class Implementation:
    def __init__(self, verbose=False):
        self.verbose = bool(verbose)
    def train(self, X):
        raise NotImplementedError(f'the train method on {self.__class__} has not been implemented')
    def fit(self, X):
        raise NotImplementedError(f'the fit method on {self.__class__} has not been implemented')
    def load_data(self, root: Path, xy: Tuple[int, int] = None):
        """
        Load ORL (or Extended YaleB) dataset to numpy array.

        :param root: root directory of dataset (data/ORL or data/CroppedYaleB)
        :param xy: requested width and height of the images.
        """

        # sanitize input
        root = Path(root)
        assert root.is_dir(), f'the directory {root} does not exist. (it may exist but is not a directory). Cannot load dataset.'

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

                # apply preprocessing
                img = self.preprocess(img)

                # convert image to numpy array.

                # collect data and label.
                images.append(img.reshape((-1,1)))
                labels.append(person_no)

        # concate all images and labels.
        print(len(images))
        images = np.concatenate(images, axis=1)
        labels = np.array(labels)

        return images.T, labels

    def preprocess(self, img):
        return img
