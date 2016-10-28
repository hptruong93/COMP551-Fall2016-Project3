from main import load_dataset
import numpy as np
from models.mnist.train_mnist import load_predictor
from __future__ import division
X_train, y_train, X_val, y_val = load_dataset()
mnist_preds_file = 'data/mnist_predictions_train.txt'
def get_subimages(image):
    # grab all the 28x28 subimages, stepping along by 2 pixels
    #   32 becaause 32 = 60 - 28
    #   this creates 32*32/(step**2) subimages
    sub_images = np.array(
            [image[i:i+28,j:j+28]
                for i in range(0,32,step) for j in range(0,32,step)]
    )
    return sub_images
mnist_preds_file = 'data/mnist_stp4_predictions_train.txt'
mnist_predictor = load_predictor()
step = 4
num_si = 32**2 // step**2
images = X_train.reshape(len(X_train),60,60)
sub_images = map(get_subimages, images)
shaper = lambda l: np.array(l).reshape(num_si*len(l), 1, 28, 28)
with open(mnist_preds_file, 'a+') as f:
    for i in xrange(len(images)//500):
        print "On batch {}/{}".format(i, len(images)//500)
        ims = images[i*500:(i+1)*500]
        sub_images = shaper(map(get_subimages, ims))
        np.savetxt(f, mnist_predictor(sub_images))
images = X_val.reshape(len(X_val),60,60)
mnist_preds_file = 'data/mnist_stp4_predictions_val.txt'
with open(mnist_preds_file, 'a+') as f:
    for i in xrange(len(images)//500):
        print "On batch {}/{}".format(i, len(images)//500)
        ims = images[i*500:(i+1)*500]
        sub_images = shaper(map(get_subimages, ims))
        np.savetxt(f, mnist_predictor(sub_images))
