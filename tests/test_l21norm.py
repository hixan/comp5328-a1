from NMF_Implementation import L21Norm
from NMF_Implementation.base import load_data

def test_fit():
    alg = L21Norm.Algorithm(15)
    X, Y = load_data('data/ORL')
    alg.fit(X)
