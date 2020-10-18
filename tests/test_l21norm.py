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
    
