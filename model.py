import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, conv2d, max_pool2d

def small_model(X,Y):
	X = tf.reshape(X, [-1,32*32])
	Y = tf.reshape(Y, [-1,2])

	logits = fully_connected(X,2, activation_fn=None)

	probs = tf.nn.softmax(logits)

	loss = tf.losses.log_loss(probs, Y)

	optimizer = tf.train.AdamOptimizer().minimize(loss)

	acc = tf.contrib.metrics.accuracy(tf.greater(Y, tf.constant(0.5)), tf.greater(probs, tf.constant(0.5)))

	return [optimizer, loss, acc, probs]


def small_conv_model(X,Y):
	X = tf.reshape(X, [-1,32,32,1])
	Y = tf.reshape(Y, [-1,2])

	cv1 = conv2d(X, 1, [5,5], stride=2)

	lin1 = tf.reshape(cv1, [-1,256])

	logits = fully_connected(lin1,2, activation_fn=None)

	probs = tf.nn.softmax(logits)

	loss = tf.losses.log_loss(probs, Y)

	optimizer = tf.train.AdamOptimizer().minimize(loss)

	acc = tf.contrib.metrics.accuracy(tf.greater(Y, tf.constant(0.5)), tf.greater(probs, tf.constant(0.5)))

	return [optimizer, loss, acc, probs]	

def fully_conv_model(X,Y):
	X = tf.reshape(X, [-1,32,32,1])
	Y = tf.reshape(Y, [-1,2])

	cv1 = conv2d(X, 2, [7,7], stride=3)
	cv2 = conv2d(cv1, 2, [5,5], stride=3)
	cv3 = conv2d(cv2, 2, [3,3], stride=1)
	cv4 = conv2d(cv3, 2, [3,3], stride=1)
	pool1 = max_pool2d(cv4, [4,4])

	cv_reshape = tf.reshape(pool1, [-1,2])

	probs = tf.nn.softmax(cv_reshape)

	loss = tf.losses.log_loss(probs, Y)

	optimizer = tf.train.AdamOptimizer().minimize(loss)

	acc = tf.contrib.metrics.accuracy(tf.greater(Y, tf.constant(0.5)), tf.greater(probs, tf.constant(0.5)))

	return [optimizer, loss, acc, probs]


	


