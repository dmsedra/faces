from load_data import DataGenerator
import numpy as np
import model
import tensorflow as tf

def main(epochs=100):
	params = {"faces_dir":"data/orl_faces", "cifar_file":"data/cifar_10/data_batch_1"}
	dg = DataGenerator(**params)
	gen = dg.batch()

	X = tf.placeholder(tf.float32, [None, 32,32])
	Y = tf.placeholder(tf.float32, [None,2])
	optimizer,cost,acc,probs = model.small_conv_model(X,Y)

	steps = int(np.ceil(dg.X.shape[0]/dg.batch_size)*epochs)
	#init = tf.global_variables_initializer()
	init = tf.global_variables_initializer()

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)

		for i in range(steps):
			X_batch, Y_batch = next(gen) #np.concatenate((np.ones((1,250,250)),np.zeros((1,250,250)))), [[0,1],[1,0]]

			#update params
			p,c,a = sess.run([optimizer,cost,acc], feed_dict={X:X_batch, Y:Y_batch})
			print("Step %d Cost %f Accuracy %f"%(steps,c,a))
			
			#evaluate performance
			if i%10 == 0:
				X_eval,Y_eval = dg.eval()
				eval_a = sess.run([acc], feed_dict={X:X_batch, Y:Y_batch})
				print("Eval Accuracy %f"%eval_a[0])

		#saver.save(sess, "models/small/model.ckpt")

if __name__ == "__main__":
	main()