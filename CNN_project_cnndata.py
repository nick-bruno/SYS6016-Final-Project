import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Step 8: Read in data
x_all = pd.read_csv('X_cnn.csv')
x_all = x_all[['0', '1', '2', '3', '4']]
y_all = pd.read_csv('Y_cnn.csv')
y_all = y_all[['0', '1']]
x_train = x_all.sample(n=int(0.80*(len(x_all))))
y_train = y_all.take(list(x_train.index))
x_valid = x_all.drop(x_train.index)
y_valid = y_all.drop(x_train.index)
x_train = x_train.reset_index(drop=True)
x_valid = x_valid.reset_index(drop=True)
y_valid = y_valid.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

# Step 1: Define parameters for the CNN

# Input
height = 1
width = 5
channels = 1
n_inputs = height * width

# Parameters for TWO convolutional layers: 
conv1_fmaps = 10 #made this up
conv1_ksize = 2
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 10 # made this up
conv2_ksize = 2
conv2_stride = 1
conv2_pad = "SAME"

# Define a pooling layer
pool3_dropout_rate = 0.25
pool3_fmaps = conv2_fmaps

# Define a fully connected layer 
n_fc1 = 128
fc1_dropout_rate = 0.5

# Output
n_outputs = 2

#reset_graph()

# Step 2: Set up placeholders for input data
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, 5], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None, 2], name="y")
    training = tf.placeholder_with_default(False, shape=[], name='training')

# Create TensorBoard Graph
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
    
# Step 3: Set up the two convolutional layers using tf.layers.conv2d
conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")

# Step 4: Set up the pooling layer with dropout using tf.nn.max_pool 
with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 1 * 2])# she had half for both,so i did floor of half
    pool3_flat_drop = tf.layers.dropout(pool3_flat, pool3_dropout_rate, training=training)

# Step 5: Set up the fully connected layer using tf.layers.dense
with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1") 
    fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training=training)


# Step 6: Calculate final output from the output of the fully connected layer
with tf.name_scope("output"):
    logits = tf.layers.dense(fc1_drop, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

# Step 5: Define the optimizer; taking as input (learning_rate) and (loss)
with tf.name_scope("train"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

# Step 6: Define the evaluation metric
#with tf.name_scope("eval"):
#    correct = tf.nn.in_top_k(logits, y, 1)
#    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#    
with tf.name_scope("eval"):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Step 7: Initiate    
with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    

# Step 9: Define some necessary functions
def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

   
def shuffle_batch(X, y, batch_size):
    batch = X.sample(n=batch_size)
    X_batch = batch.iloc[:,0:].values
    X_batch = X_batch.astype(np.float32)
    y_batch = y.take(list(batch.index))
    yield X_batch, y_batch
    

# Step 10: Define training and evaluation parameters
n_epochs = 2000
batch_size = 50
iteration = 0

best_loss_val = np.infty
check_interval = 500
checks_since_last_progress = 0
max_checks_without_progress = 20
best_model_params = None 

# Step 11: Train and evaluate CNN with Early Stopping procedure defined at the very top
with tf.Session() as sess:
    init.run()
    trainingAcc = []
    validAcc = []
    epochs = []
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(x_train, y_train, batch_size):
            iteration += 1
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
            if iteration % check_interval == 0:
                loss_val = loss.eval(feed_dict={X: x_valid, y: y_valid})
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    checks_since_last_progress = 0
                    best_model_params = get_model_params()
                else:
                    checks_since_last_progress += 1
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: x_valid, y: y_valid})
        if epoch%25==0:
            trainingAcc.append(acc_batch)
            epochs.append(epoch)
            validAcc.append(acc_val)
        print("Epoch {}, last batch accuracy: {:.4f}%, valid. accuracy: {:.4f}%, valid. best loss: {:.6f}".format(
                  epoch, acc_batch * 100, acc_val * 100, best_loss_val))
        if checks_since_last_progress > max_checks_without_progress:
            print("Early stopping!")
            break

# Making plots
df=pd.DataFrame({'epoch': epochs, 'TrainingAccuracy': trainingAcc, 'ValidationAccuracy':validAcc})
plt.plot( 'epoch', 'TrainingAccuracy', data=df, color='skyblue', linewidth=2)
plt.plot( 'epoch', 'ValidationAccuracy', data=df, marker='', color='olive', linewidth=2)
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy for CNN')
plt.legend()
plt.show()
   
writer.close()
