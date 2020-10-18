from NMF_Implementation import L21Norm
from NMF_Implementation.base import load_data
from matplotlib import pyplot as plt
import numpy as np


def test_fit():
    alg = L21Norm.Algorithm(15)
    X, Y = load_data('data/ORL')
    X = X[:5]

    # should take in rows of images
    alg.fit(X)
    # transformation sholud work with the same shape
    trans = alg.transform(X)

    # show first 5 components
    for c in alg.transform(np.eye(15)[:5]):
        plt.imshow(c)
        plt.show()
    

def random(k, n, p, tolerance=None):
    k = 15
    n = 300
    p = 30

    print(f"k={10}\nn={300}\np={15}")
    alg = L21Norm.Algorithm(k)
    data = np.random.randint(0, 255, (n, p))
    alg.fit(data)
    assert alg.transform(data).shape == (n, k)
    assert alg.inverse_transform(np.random.randint(0, 255, (2*n, k))).shape == (2*n, k)

    if tolerance is not None:
        # max deviation
        maxdev = np.max(alg.inverse_transform(alg.transform(data)) - data)
        print('Maximum difference', maxdev)
        assert maxdev <= tolerance


def test_random_shapes():
    random(2, 50, 8)


def test_random_values():
    random(8, 50, 8, 0.01)
