from NMF_Implementation.base import load_data
from collections import Counter
from my_tools.tools import inspect_array
import matplotlib.pyplot as plt
import numpy as np


def gen_bool_index_random(arr, p):
    n = np.product(arr.shape[1:])
    bools = np.tile(np.concatenate((np.ones(round(n * p)),
        np.zeros(round(n - n * p)))).astype(bool), (arr.shape[0], 1))
    for i in bools:
        np.random.shuffle(i)
    return bools


def salt_and_pepper(X, p, r, salt=255, pepper=0):
    """add salt and pepper noise to a dataset of arrays.

    :param X: input data. Should take the shape of (n samples, *dimensions)
    :param p: proportion of features to change.
    :param r: proportion of changed features to change to salt
    :param salt: value to replace with if designated to salt
    :param min: value to replace with if designated to pepper
    :return: X, but with p*r*100 % salt and p*(1-r)*100 % pepper.
    """

    X = X.copy()
    shape = X.shape[1:]
    n = np.product(shape)
    X = np.reshape(X.copy(), (-1, n))  # dont overwrite original
    # should this index change?
    bools_change = gen_bool_index_random(X, p)

    # the positions of changed indexes (image number, pixel number)
    im, px = np.where(bools_change)
    indecies = px.reshape(X.shape[0], -1)
    changed = indecies.shape[1]  # number of changed pixels
    for a in indecies:
        # randomize order for each image
        np.random.shuffle(a)
    px = indecies.flatten()
    arr_indexes = np.column_stack((im, px)).T
    new_values = np.tile(
            np.concatenate((np.ones(round(changed * r)) * salt,
                np.ones(changed - round(changed * r)) * pepper)),
            X.shape[0]
    )
    X[arr_indexes[0], arr_indexes[1]] = new_values

    return np.reshape(X, (-1, *shape))


def missing_grid(X, grid_spacing, grid_size=(0,0), grid_colors=(0,0), grid_offset=(0,0)):
    X = X.copy()
    assert len(X.shape) == 3, 'this function must recieve images as matrices instead of vectors'
    xticks = np.arange(grid_offset[0], X.shape[1], grid_spacing[0])
    yticks = np.arange(grid_offset[1], X.shape[2], grid_spacing[1])
    for i in range(grid_size[0]):
        idxs = xticks + i
        idxs = idxs[idxs+i <= X.shape[1]]
        X[:, idxs, :] = grid_colors[0]

    for i in range(grid_size[1]):
        idxs = yticks + i
        idxs = idxs[idxs+i <= X.shape[2]]
        X[:, :, idxs] = grid_colors[1]
    return X


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
