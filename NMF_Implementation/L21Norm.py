import numpy as np

from .base import Implementation

class Algorithm(Implementation):

    def __init__(self, components, max_iter=100, initial_dictionary=None, image_shape=None):
        if initial_dictionary is not None:
            initial_dictionary = initial_dictionary.copy()
        self._metavalues = dict(
            name = 'L2,1 Norm NMF',
            training_loss = [],
            training_residue = [],
            components = components,
            max_iter = max_iter,
            initial_dictionary = initial_dictionary,
            image_shape = image_shape,
        )
        self._dictionary = None

    def fit(self, X: np.ndarray, initial_representation=None):

        if self._metavalues['image_shape'] is None:
            # initialise default image shape if was not previously assigned
            self._metavalues['image_shape'] = X.shape[1:]
        else:
            # sanity checks
            assert X.shape[1:] == self._metavalues['image_shape'], ('input data does '
           'not match expected shape')

        # reshape the data to be vectors instead of images (if not already reshaped)
        n: int = X.shape[0]
        p: int = np.product(X.shape[1:])
        k: int = self._metavalues['components']
        X: np.ndarray = np.reshape(X, (-1, p)).T

        # n - number of input images
        # p - dimensionality of population space
        # k - number of components

        # X shape (p, n)
        # D shape (p, k)
        # R shape (k, n)

        # initialise the learning dictionary if not already initialised
        if self._metavalues['initial_dictionary'] is None:
            self._metavalues['initial_dictionary'] = np.random.rand(p, k)
        else:
            assert self._metavalues['initial_dictionary'].shape == (p, k)

        # initialize dictionary if not already done.
        if self._dictionary is None:
            self._dictionary = self._metavalues['initial_dictionary'].copy()

        # initialize representation
        if initial_representation is None:
            R: np.ndarray = np.random.rand(k, n)
        else:
            R: np.ndarray = initial_representation.copy()
            assert R.shape == (k, n)

        D: np.ndarray = self._dictionary  # alias for readability.

        # toggle optimizing between D and R
        optim = 'D'

        # marker for different calls
        self._metavalues['training_loss'].append(None)
        self._metavalues['training_residue'].append(None)

        # fit the data
        for iteration in range(self._metavalues['max_iter'] * 2):  # *2 to account for alternation
            # this section follows section 2.7 of the accompanied documentation in
            # ../papers/Robust Nonnegative Matrix Factorization using L21 Norm 2011.pdf
            diffs = X - D @ R
            loss = l21_norm(diffs)
            residue = np.linalg.norm(diffs)

            # keep these for later
            self._metavalues['training_loss'].append(loss)
            self._metavalues['training_residue'].append(residue)

            # TODO set optim='stop' based on some condition on the loss / residue.

            diag = np.diag(1 / np.linalg.norm(diffs, axis=0))

            if optim == 'D':
                optim = 'R'  # toggle for next time
                D *= (X @ diag @ R.T) / (D @ R @ diag @ R.T)
            elif optim == 'R':
                optim = 'D'
                R *= (D.T @ X @ diag) / (D.T @ D @ R @ diag)

            elif optim == 'stop':
                break
            else:
                assert 0, 'optim not recognised'

    def transform(self, X):
        pass

    def inverse_transform(self, R):
        pass

    def get_metavalues(self):
        """ L21Norm.Algorithm.get_metavalues

        returns a dict with the following attributes:
        'name' : name of the algorithm
        'training_loss' : loss of the algorithm at each iteration during training
        'components' : dimensionality of the representation vectors.
        'max_iter' : maximum number of iterations in training
        """
        return self._metavalues

def l21_norm(arr):
    return np.sum(np.linalg.norm(arr, axis=1))
