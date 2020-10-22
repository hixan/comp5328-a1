import numpy as np

from .base import Implementation


class Algorithm(Implementation):

    def __init__(self, components, stop_threshold=0.01, max_iter=200,
            initial_dictionary=None, image_shape=None, stop_window=10):
        """ L2Norm NMF

        :param components: number of components to train
        :param stop_threshold: threshold under which the improvement (%) is
            determined not worth it.
        :param max_iter: maximum number of iterations (if stop_threshold does
            not trigger)
        :param initial_dictionary: Dictionary to use as initialization. None
            indicates a random dictionary will be used
        :param image_shape: shape of input images/matrices (if None it will be
            implied from the data on first fit.
        """

        if initial_dictionary is not None:
            initial_dictionary = initial_dictionary.copy()
        self._metavalues = dict(
            name='L2 Norm NMF',
            training_loss=[],
            training_residue=[],
            components=components,
            stop_threshold=stop_threshold,
            stop_window=stop_window,
            max_iter=max_iter,
            initial_dictionary=initial_dictionary,
            image_shape=image_shape,
        )
        self._dictionary = None
        self._inverse_dictionary = None

    def fit(self, X: np.ndarray, initial_representation=None):
        """ Assumes first dimension of X represents rows of data """

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
        X: np.ndarray = Algorithm._reshape_forward(X)
        assert X.shape == (p, n)

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
        # start with updating 'R'
        optim = 'R'

        # marker for different calls
        self._metavalues['training_loss'].append(None)
        self._metavalues['training_residue'].append(None)

        # fit the data
        for iteration in range(self._metavalues['max_iter'] * 2):  # *2 to account for alternation
            # this section follows section 2.7 of the accompanied documentation in
            # ../papers/Robust Nonnegative Matrix Factorization using L21 Norm 2011.pdf

            diffs = X - D @ R

            if optim == 'D':
                optim = 'R'  # toggle for next time
                D *= (X @ R.T) / (D @ R @ R.T)
            elif optim == 'R':
                optim = 'D'
                R *= (D.T @ X) / (D.T @ D @ R)

            elif optim == 'stop':
                break
            else:
                assert 0, 'optim not recognised'

            # only collect the loss after D has been updated
            if optim == 'R':  # Actually activates on 'D' pass, as flag is changed above.
                # average loss per example
                loss = l2_norm(diffs) / n
                residue = np.linalg.norm(diffs)

                # keep these for later
                self._metavalues['training_loss'].append(loss)
                self._metavalues['training_residue'].append(residue)

                # computing if stopping condition is met
                if iteration // 2 - 1 > self._metavalues['stop_window']:
                    previous_loss = self._metavalues['training_loss'
                            ][-self._metavalues['stop_window']]
                    current_loss = self._metavalues['training_loss'][-1]
                    relative_improvement = ((previous_loss - current_loss) /
                           self._metavalues['stop_window'] / previous_loss)
                    if relative_improvement < self._metavalues['stop_threshold']:
                        optim = 'stop'

        self._inverse_dictionary = np.linalg.inv(D.T @ D) @ D.T

    def transform(self, X):
        """ Transform X into its representation

        :param X: row matrix/tensor of same shape as training time representing
            data. If there are n images of size 10x5, X should be of shape
            (n, 10, 5) or (n, 50) (depending on what was passed at training
            time)
        :return: row matrix (n, k) representing X.

        Returns a row oriented matrix of representation vectors of X
        """
        return (self._inverse_dictionary @ Algorithm._reshape_forward(X)).T

    def inverse_transform(self, R):
        """ Transform representations of X back into X.

        :param R: row oriented matrix of representation vectors
        :return: row matrix/tensor of the same shape as input (barring first
        dimension)
        """
        return self._reshape_backward(self._dictionary @ R.T)

    def get_metavalues(self):
        """ L21Norm.Algorithm.get_metavalues

        returns a dict with the following attributes:
        'name' : name of the algorithm
        'training_loss' : loss of the algorithm at each iteration during
            training
        'components' : dimensionality of the representation vectors.
        'max_iter' : maximum number of iterations in training
        """
        return self._metavalues

    @staticmethod
    def _reshape_forward(mat):
        """ transpose a row matrix or tensor to a column matrix """
        if len(mat.shape) == 3:
            return mat.reshape(mat.shape[0], -1).T
        if len(mat.shape) == 2:
            return mat.T
        raise ValueError(f'expected a 2 or 3 dimensional matrix. Got a matrix '
                         'of shape {mat.shape}')

    def _reshape_backward(self, mat):
        """transpose a column matrix to a row matrix / tensor (dependant on
        input to this class on training)"""
        assert len(mat.shape) == 2, 'needs a matrix not a tensor'
        newshape = self._metavalues['image_shape']
        return mat.T.reshape(mat.shape[1], *newshape)


def l2_norm(arr):
    return np.linalg.norm(arr)
