from NMF_Implementation.base import Implementation, load_data
show_results = False

def test_load_data_ORL():
    """test_load_data_ORL.
    the dataset should have 40 subjects, each with 10 faces. (total of 400 observations)
    This test loads the data and checks that the correct amount of data is contained.
    """
    x, y = load_data('data/ORL')
    assert x.shape[0] == 400
    assert y.shape == (400,)

def test_load_data_resize():
    """test_load_data_resize.
    checks that the faces are being resized at all (no check to integrity)
    """
    x, y = load_data('data/ORL', (3, 3))
    assert x.shape[1:] == (3, 3)

def test_load_data_CroppedYale():
    """test_load_data_CroppedYale.

    should have 2414 observations from 38 participants.
    """
    x, y = load_data('data/CroppedYaleB', (3, 3))
    print(x.shape)
    assert x.shape[0] == 2414
    assert x.shape[1:] == (3, 3)

