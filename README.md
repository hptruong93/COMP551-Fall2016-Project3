# COMP551-Fall2016-Project3
COMP551-Fall2016-Project3 Difficult digits

[Kaggle](https://inclass.kaggle.com/c/difficult-digits-fall-2016/)

[Report](https://www.sharelatex.com/project/581a00b68bc686ac2b6baaf2)

### Usage Notes
The data directory contains all the data files from the kaggle. They have been omitted from the git due to their size,
but should be added back for our code to properly run

#### Logistic Regression

Run `python main.py lr` to begin a crossvalidation of hyperparamter settings.


#### Feedforward network implementation

From models/ann_imp run `python ann.py e l1 l2 l3 ln` where e is the number of epochs, l1, l2, ..., ln are the sizes of the hidden layers of the network.
Sample code to produce prediction is also provided at the bottom of this file (the commented code)

#### CNN

Run `python main.py cnn` to begin training 100 epochs of our CNN. The number of epochs can be changed by editing `main.py`, line 65.
Note that this prints to both stdout and stderr, so you can redirect the log to a file and you will still see output.

#### Lenet

From models run `python lenet.py`. This runs the current configuration of Lenet. To modify the configuration of the network inside the file, modify the parameters to the call
to`evaluate_lenet5` method in `main`

#### Making predictions for a CNN

Models are saved in `.npz` files in the models directory. Predictions can be made using an ipyton session like the following one:

```python
from main import *
from models.basic_cnn.cnn import *

input_var = T.tensor4('inputs')
network = build_cnn(input_var)

with np.load('models/basic_cnn/basic_cnn_model_70s.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(network, param_values)

X = np.fromfile('data/test_x.bin', dtype='uint8')
X = X.reshape(-1,1,60,60)
X = np.apply_along_axis(lambda im: (im > 252).astype(np.float32), 0, X)
prediction = lasagne.layers.get_output(network, deterministic=True)
predict_fcn = theano.function([input_var], prediction)
def predictor(images):
    results = predict_fcn(images)
    return map(np.argmax, results)

y = []
# incrementally to avoid memory concerns
for i in xrange(0, len(X), 500):
    ims = X[i:i+500]
    y.append(predictor(ims))

# flatten
y = [ i for sl in y for i in sl]

import csv
 with open('cnn_prediction.csv','w+') as f:
    w = csv.writer(f)
    w.writerow(['Id','Prediction'])
    for r in enumerate(y):
        w.writerow(r)
```

