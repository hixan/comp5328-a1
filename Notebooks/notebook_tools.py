from NMF_Implementation.Noise import reconstruction_error_procedure
from matplotlib import pyplot as plt


def reconstruction_error_sample(X, modelfactory, noisefunc, sample_size=5, return_model=False):
    rv = []
    for run in range(sample_size):
        # observe one experimental result 5 times
        model = modelfactory()
        # save them seperately
        rv.append(dict(
            reconstruction_error=reconstruction_error_procedure(
            X, 0.9, model, noisefunc),
            run=run,
            iterations=len(model.get_metavalues()['training_loss'])
        ))
        # the garbage collecter sometimes seems to miss this until
        # later, explicitly free the memory now.
        
    # return list of runs
    if return_model:
        return rv, model
    return rv

def show_images(imagearray, title, show_image_dims):
    plt.suptitle(title, y=1.2)
    for i, image in enumerate(imagearray):
        plt.subplot(*show_image_dims, i+1)
        plt.imshow(image, cmap='Greys_r')
        plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    
