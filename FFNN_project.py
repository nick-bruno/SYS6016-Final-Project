
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

os.chdir('C:\\Users\\cathe\\Desktop\\DeepLearning')

x = pd.read_csv('X.csv').astype('float32')
x = x[['AAPL_pct_change', 'AMZ_pct_change', 'GOOG_pct_change', 'CSCO_pct_change', 
       'MSFT_pct_change', 'FB_pct_change', 'IBM_pct_change', 'time']]
x = x.iloc[0:421]
y = pd.read_csv('IBM_y.csv')
y = y[['IBM_y','IBM_y2']]

x_train = x.sample(n=int(0.80*(len(x))))
y_train = y.take(list(x_train.index))
x_valid = x.drop(x_train.index)
y_valid = y.drop(x_train.index)

n_inputs = x.shape[1]  
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 2

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
Y = tf.placeholder(tf.int32, shape=(None, n_outputs), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                              activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                              activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")


writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.name_scope("loss"):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits)
    loss_summary = tf.summary.scalar('log_loss', loss)
    
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
   
with tf.name_scope("eval"):
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(logits, 1))
    accuracy = 100 * tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 100
batch_size = 50

with tf.Session() as sess:
    init.run()
    train_acc_list = []
    val_acc_list = []
    loss_list = []
    tot_loss = 0
    epochs = []
    for epoch in range(n_epochs):
        remaining = x
        remaining = remaining.reset_index(drop=True)
        for iteration in range(len(x) // batch_size):
            batch = remaining.sample(n=batch_size)
            X_batch = batch.iloc[:,0:].values
            X_batch = X_batch.astype(np.float32)
            y_batch = y.take(list(batch.index))
            remaining = remaining.drop(batch.index)
            sess.run(training_op, feed_dict = {X:X_batch, Y:y_batch})
            tot_loss += sess.run(loss, feed_dict={X: X_batch, Y: y_batch})
        acc_train = accuracy.eval(feed_dict = {X:X_batch, Y:y_batch})
        train_acc_list.append(acc_train)
        acc_val = accuracy.eval(feed_dict= {X: x_valid, Y: y_valid})
        val_acc_list.append(acc_val)
        loss_list.append(tot_loss)
        epochs.append(epoch)
        print(epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val)
    save_path = saver.save(sess, "./my_model_final4.ckpt")


with tf.Session() as sess:
    saver.restore(sess, "./my_model_final4.ckpt")
    Z = logits.eval(feed_dict = {X: x_valid})
    y_pred = np.argmax(Z, axis = 1)

print("Predicted classes:", y_pred)
#print("Actual classes:", np.argmax(y_valid))

df=pd.DataFrame({'epoch': epochs, 'TrainingAccuracy': train_acc_list, 'ValidationAccuracy':val_acc_list})
plt.plot( 'epoch', 'TrainingAccuracy', data=df, color='skyblue', linewidth=2)
plt.plot( 'epoch', 'ValidationAccuracy', data=df, marker='', color='olive', linewidth=2)
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy for FFNN')
plt.legend()
plt.show()

writer.close()