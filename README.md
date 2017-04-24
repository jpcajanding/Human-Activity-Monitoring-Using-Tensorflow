# Human-Activity-Monitoring-Using-Tensorflow

This is an implemetation of a two layer feed forward neural network in tensorflow to classify human activity as defined by the propenents of the Human Activity Monitoring dataset from UCI.

To run this code, you first need to download the UCI dataset (https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones) and save it to a folder named 'data' on the program path.

This implementation has 2 layers, with each layer having 561 hidden units. Training runs on a minibatch of 128 inputs and stops after 500 epochs. Drop out is implemented during training at a keep rate of 80%

The code is implemented on both python script and a notebook.

Again, this is my training on coding neural networks in tensorflow. So I appreciate all bugs and improvements reported.

This code has a test accuracy of 94%.

Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.
