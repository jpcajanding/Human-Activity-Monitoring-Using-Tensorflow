import tensorflow as tf
import numpy as np
import pandas as pd
import os
import pickle

from sklearn.utils import shuffle
from sklearn import preprocessing

testfile = r'data\test.pkl'
trainfile = r'data\train.pkl'

# preprocess labels to convert strings to numbers
le = preprocessing.LabelEncoder()

if not os.path.isfile(testfile):
    # read data
    test = pd.read_csv(r'data\test.csv')

    # shuffle data
    test  = shuffle(test)
    test_data = test.drop(['Activity','subject'] , axis=1).values
    test_label_raw = test.Activity.values
    le.fit(test_label_raw)
    test_label = le.transform(test_label_raw)

    with open(testfile,'wb') as fp:
        pickle.dump([test_label,test_data],fp,-1)
        fp.close()

if not os.path.isfile(trainfile):
    # read data
    train = pd.read_csv(r'data\train.csv')

    # shuffle data
    train = shuffle(train)
    train_data = train.drop(['Activity','subject'] , axis=1).values
    train_label_raw = train.Activity.values
    le.fit(train_label_raw)
    train_label = le.transform(train_label_raw)

    with open(trainfile,'wb') as fp:
        pickle.dump([train_label,train_data],fp,-1)
        fp.close()

#load data
with open(testfile, 'rb') as fp:
    test_label, test_data = pickle.load(fp)
    test_data = test_data.astype('float32')
    fp.close()

with open(trainfile, 'rb') as fp:
    train_label, train_data = pickle.load(fp)
    fp.close()

#define ANN architecture
n_train, n_input = np.shape(train_data)
n_test = test_label.size

num_labels = 6
train_labels = np.zeros((n_train,num_labels))
train_labels[np.arange(n_train),train_label] = 1.0
test_labels = np.zeros((n_test,num_labels))
test_labels[np.arange(n_test),test_label] = 1.0

#set hyperparameters
epochs = 500
batch_size = 128
nhidden1 = 561
nhidden2 = 561
learning_rate = 0.001
dropout = 0.8
n_out = 6
num_steps = int((n_train/batch_size)*epochs)

graph = tf.Graph()
with graph.as_default():
    #I really don't know why I need to graph

    #training placeholders
    tf_train_in = tf.placeholder(tf.float32,shape=(None,n_input))
    tf_train_label = tf.placeholder(tf.float32,shape=(None,n_out))

    #input to hidden layer
    weights_in = tf.Variable(tf.truncated_normal([n_input,nhidden1]))
    bias_in = tf.add(tf.Variable(tf.zeros(nhidden1)),0.1)

    #hidden1 layer to hidden2 layer
    weights_h = tf.Variable(tf.truncated_normal([nhidden1,nhidden2]))
    bias_h = tf.add(tf.Variable(tf.zeros(nhidden2)),0.1)

    #hidden2 layer to output
    weights_out = tf.Variable(tf.truncated_normal([nhidden2,n_out]))
    bias_out = tf.add(tf.Variable(tf.zeros(n_out)),0.1)

    #model definition
    def ANN_model(data, dropout = 0.5):
        logits_h1 = tf.add(tf.matmul(data,weights_in),bias_in)
        relu_h1 = tf.nn.relu(logits_h1)
        drop_h1 = tf.nn.dropout(relu_h1,dropout)

        logits_h2 = tf.add(tf.matmul(drop_h1,weights_h),bias_h)
        relu_h2 = tf.nn.relu(logits_h2)
        drop_h2 = tf.nn.dropout(relu_h2,dropout)

        logits = tf.add(tf.matmul(drop_h2,weights_out),bias_out)

        return logits, tf.nn.softmax(logits)

    train_logits, train_pred = ANN_model(tf_train_in,dropout= dropout)

    #define loss function, optimizer, cost
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train_logits, labels = tf_train_label))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    test_logits, test_pred = ANN_model(test_data, dropout= 1.0)

def accuracy(predictions, labels):
    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(predictions, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy.eval()*100.0

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print(tf.__version__)
    for step in range(num_steps):
        offset = (step * batch_size)% (n_train - batch_size)
        # Generate a minibatch.
        batch_data = train_data[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size)]
        feed_dict = {tf_train_in:
                     batch_data, tf_train_label: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_pred], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch (size=%d) loss at step %d: %f" % (batch_size, step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions,
                                                          batch_labels))
    # Accuracy: 91.6%
    print("Test accuracy: %.1f%%" % accuracy(test_pred.eval(),
                                             test_labels))


