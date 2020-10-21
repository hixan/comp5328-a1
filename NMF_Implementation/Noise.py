import numpy as np
import math


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


def missing_rects(X, rect_size, nrow=2, ncol=2):

    # construct a grid of what to show and use it as a mask for the image.
    offset = 0, 0  # TODO this should also be calculated to handle rounding errors
    # an additional mask will also have to be applied
    grid_size = (
        int((X.shape[1] - ncol * rect_size[0]) / (ncol + 1)),
        int((X.shape[2] - nrow * rect_size[1]) / (nrow + 1))
    )
    grid_spacing = (
        int((X.shape[1] - grid_size[0]) / ncol),
        int((X.shape[2] - grid_size[1]) / nrow)
    )
    return X.copy() * missing_grid(
            np.zeros(X.shape),
            grid_spacing=grid_spacing,
            grid_size=grid_size,
            grid_colors=(True, True),
            grid_offset=offset
    ).astype(int)


def reconstruction_error_procedure(X, fraction, algorithm, noise_func):

    # total number of examples
    total_num_examples = X.shape[0]

    # number of examples for the random subset
    num_examples = math.floor(fraction * total_num_examples)

    # draw indices of example for the random subset
    indices = np.random.choice(np.array(range(total_num_examples)), num_examples, replace=False)

    # generate subset
    subset = X[indices]

    # add salt and pepper noise
    noisy_subset = noise_func(subset)

    # fit on the noisy subset
    algorithm.fit(noisy_subset)

    # compute reconstruction error
    reconstruction_error = algorithm.reconstruction_error(noisy_subset, subset)

    return reconstruction_error
