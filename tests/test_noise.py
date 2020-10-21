from NMF_Implementation.base import load_data
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from NMF_Implementation.Noise import salt_and_pepper


def test_salt_and_pepper():
    # white image, grey image, black image (all 16x16
    X = (np.ones((3, 16, 16)) * np.tile(np.array([[[0, 128, 255]]]).T, (1,16,16))).astype(np.uint8)
    # >>> X.shape
    # <<< (3, 16, 16)
    # >>> np.sum(X[0])
    # <<< 0
    # >>> Counter(X[1].flatten())
    # <<< Counter({128: 256})
    # >>> Counter(X[2].flatten())
    # <<< Counter({255: 256})

    def makeup(images):
        # returns a counter for each element of the outer array (see comments above)
        return np.sum(np.apply_along_axis(Counter, 1, images), axis=tuple(range(1, len(images.shape)-1)))

    orig_makeup = makeup(X)
    # all only have one element, and those elements are 0, 128, 255
    assert np.all(np.array(list(map(lambda x: len(x.keys()), orig_makeup))) == 1)
    assert list(map(lambda x: next(iter(x.keys())), orig_makeup)) == [0, 128, 255]

    noise_x = salt_and_pepper(X, p=.2, r=.5)

    # all white
    print('all white')
    noise_x = salt_and_pepper(X, p=1.0, r=1.0)
    assert noise_x.shape == X.shape
    new_makeup = makeup(noise_x)
    assert np.all(np.array(list(map(lambda x: len(x.keys()), new_makeup))) == 1)
    assert list(map(lambda x: next(iter(x.keys())), new_makeup)) == [255, 255, 255]

    print()

    # all black
    print('all black')
    noise_x = salt_and_pepper(X, p=1.0, r=0.0)
    new_makeup = makeup(noise_x)
    assert np.all(np.array(list(map(lambda x: len(x.keys()), new_makeup))) == 1)
    assert list(map(lambda x: next(iter(x.keys())), new_makeup)) == [0, 0, 0]

    print()

    # 20% white
    print('20% white')
    noise_x = salt_and_pepper(X, p=1.0, r=0.2)
    new_makeup = makeup(noise_x)
    print(*new_makeup, sep='\n')
    assert abs(new_makeup[0][255] - int(256 * .2)) < 2  # originally black
    assert abs(new_makeup[1][255] - int(256 * .2)) < 2
    assert abs(new_makeup[2][255] - int(256 * .2)) < 2  # originally white
    assert abs(new_makeup[0][0] - int(256 * .8)) < 2  # originally black
    assert abs(new_makeup[1][0] - int(256 * .8)) < 2
    assert abs(new_makeup[2][0] - int(256 * .8)) < 2  # originally white

    print()

    # 20% white, 0% black
    print('20% white, 0% black')
    noise_x = salt_and_pepper(X, p=.2, r=1.0)
    new_makeup = makeup(noise_x)
    print(*new_makeup, sep='\n')
    assert abs(new_makeup[0][255] - int(256 * .2)) < 2  # originally black
    assert abs(new_makeup[1][255] - int(256 * .2)) < 2
    assert abs(new_makeup[2][255] - int(256 * 1.0)) < 2  # originally white
    assert abs(new_makeup[0][0] - int(256 * .8)) < 2  # originally black
    assert abs(new_makeup[1][0] - int(256 * .0)) < 2
    assert abs(new_makeup[2][0] - int(256 * .0)) < 2  # originally white

    print()

    # 10% white, 30% black
    print('10% white, 30% black')
    noise_x = salt_and_pepper(X, p=.4, r=.25)  # .4*.25 = .1; .4*.75 = .3
    new_makeup = makeup(noise_x)
    print(*new_makeup, sep='\n')
    assert abs(new_makeup[0][255] - int(256 * .1)) < 2  # originally black
    assert abs(new_makeup[1][255] - int(256 * .1)) < 2
    assert abs(new_makeup[2][255] - int(256 * .7)) < 2  # originally white
    assert abs(new_makeup[0][0] - int(256 * .9)) < 2  # originally all black
    assert abs(new_makeup[1][0] - int(256 * .3)) < 2
    assert abs(new_makeup[2][0] - int(256 * .3)) < 2  # originally white
