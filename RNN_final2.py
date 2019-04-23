import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import random
import sys
sys.path.append('..')
import time
import numpy as np
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
import utils

# Start with amazon arima modeling #
# amazon = pd.read_csv('/Users/nickbruno/Downloads/amz_with_arima.csv')
# X_train = amazon[['AMZ_pct_change', 'arima', 'garch']]
# X_train = np.array(X_train)
# y_train = np.array(amazon['AMZ_y'])

#apple = pd.read_csv('appl_rnn.csv')
#apple = apple[['t1pct', 't1ar', 't1garch', 't2pct', 't2ar', 't2garch', 'AAPL_pct_change', 'arima', 'garch', 'AAPL_y']]
#X_train = np.array(apple.loc[:, apple.columns != 'AAPL_y'])
#y_train = apple['AAPL_y']
#y_train['new_column'] = 1 - apple.AAPL_y
#y_train = np.array(y_train)


apple = pd.read_csv('appl_rnn.csv')
apple = apple[['t1pct', 't1ar', 't1garch', 't2pct', 't2ar', 't2garch', 'AAPL_pct_change', 'arima', 'garch', 'AAPL_y']]
X_train = apple.loc[:, apple.columns != 'AAPL_y']
Y_train = apple['AAPL_y']
#Y_train['new_column'] = 1 - apple.AAPL_y


x_train = X_train.sample(n=int(0.80*(len(X_train))))
y_train = Y_train.take(list(x_train.index))
x_valid = X_train.drop(x_train.index)
y_valid = Y_train.drop(x_train.index)
x_train = x_train.reset_index(drop=True)
x_valid = x_valid.reset_index(drop=True)
y_valid = y_valid.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

#
#batch = pd.concat([x_train, y_train], axis=1)
#batch_sampled = batch.sample(n=10)
#X_batch = batch_sampled.iloc[:, 0:9]
#y_batch = batch_sampled.iloc[:, 9:]
#print(X_batch.head())
#print(y_batch.head())

# reset_graph()

n_steps = 3
n_inputs = 3
n_neurons = 150
n_outputs = 2

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps*n_inputs])
X_shape = tf.reshape(X, shape=[-1, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons) # can modify the activation function
outputs, states = tf.nn.dynamic_rnn(basic_cell, X_shape, dtype=tf.float32)


logits = tf.layers.dense(states, n_outputs)
# logits = tf.layers.dense(outputs, n_outputs) # one dense layer
# logits = logits.reshape(n_steps, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                          logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # can change optimizer
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
# X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
# y_train = y_train.astype(np.int32)
# y_test = y_test.astype(np.int32)
# X_valid, X_train = X_train[:5000], X_train[5000:]
# y_valid, y_train = y_train[:5000], y_train[5000:]

# Split data into training and testing #


#def shuffle_batch(X, y, batch_size):
#    rnd_idx = np.random.permutation(len(X))
#    n_batches = len(X) // batch_size
#    for batch_idx in np.array_split(rnd_idx, n_batches):
#        X_batch, y_batch = X[batch_idx], y[batch_idx]
#        yield X_batch, y_batch

# def shuffle_batch(X, y, batch_size):
#     X_batch = X_train
#     y_batch = y_train 
#     yield X_batch, y_batch

# X_test = X_test.reshape((-1, n_steps, n_inputs))
        
    
def shuffle_batch(X, y, batch_size):
    batch = X.sample(n=batch_size)
    X_batch = batch.iloc[:,0:].values
    X_batch = X_batch.astype(np.float32)
    y_batch = y.take(list(batch.index))
    yield X_batch, y_batch
    

n_epochs = 2000
batch_size = 75

with tf.Session() as sess:
    init.run()
    trainingAcc = []
    validAcc = []
    epochs = []
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(x_train, y_train, batch_size):
            #X_batch = X_batch.reshape(-1, n_steps, n_inputs)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: x_valid, y: y_valid})
        if epoch%25==0:
            epochs.append(epoch)
            trainingAcc.append(acc_batch)
            validAcc.append(acc_val)
        print(epoch, "Last batch accuracy:", acc_batch) 
        print(epoch, "Validation Accuracy:", acc_val)# "Test accuracy:", acc_test)
        
        
        
# Making plots
df=pd.DataFrame({'epoch': epochs, 'TrainingAccuracy': trainingAcc, 'ValidationAccuracy':validAcc})
plt.plot( 'epoch', 'TrainingAccuracy', data=df, color='skyblue', linewidth=2)
plt.plot( 'epoch', 'ValidationAccuracy', data=df, marker='', color='olive', linewidth=2)
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy for RNN')
plt.legend()
plt.show()