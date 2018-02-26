import numpy as np
import cv2
import tensorflow as tf
import sys
sys.path.append("../")
import model
from PIL import Image
import time

def run():
	cap = cv2.VideoCapture(0)
	cap.set(5, 5)
	tf.reset_default_graph()
	model_path = "../models/small/model.ckpt"

	X = tf.placeholder(tf.float32, [None, 32,32])
	Y = tf.placeholder(tf.float32, [None,2])
	optimizer,cost,acc,prob = model.small_model(X,Y)

	saver = tf.train.Saver()

	with tf.Session() as sess:
	    saver.restore(sess, model_path)

	    while True:
		    # Capture frame-by-frame
		    time.sleep(0.05)
		    ret, frame = cap.read()

		    # Our operations on the frame come here
		    props = getSimpleProposals(frame)
		    X_all = []
		    for region,img in props:
		        img_res = np.asarray(img.resize((32,32)).convert('1'))
		        X_all.append(img_res)
		        
		    probs_ = sess.run([prob], feed_dict={X:X_all,Y:np.zeros((len(X_all),2))})
		    probs = probs_[0]

		    max_face = max([i for i in range(len(probs)) if probs[i][0] > probs[i][1]])
			
		    a,b,size = props[max_face][0]
		    cv2.rectangle(frame, (a,b), (a+size,b+size), (0,255,0))
		    

		    # Display the resulting frame
		    cv2.imshow('frame',frame)
		    if cv2.waitKey(1) & 0xFF == ord('q'):
		        break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

def getRectangle(frame):
	x,y,_ = np.argmax(frame[:,:,0],0)	
	cv2.rectangle(frame, [x,y], [x,y]+10, [0,255,0])

def getSimpleProposals(image):
    img = Image.fromarray(image)
    h,w = img.size
    base = min(h,w)
    cur_size = 50

    props = []
    count = 0
    while cur_size <= base:
        for a in range(0,w,cur_size):
            for b in range(0,h,cur_size):
                crop = img.crop((a,b,a+cur_size,b+cur_size))
                #if np.var(np.mean(crop,2)) < 100: continue
                #crop = crop.resize((250,250))
                #crop.save("../data/thumb/%d%d%d.jpg"%(a,b,cur_size))
                
                count += 1
                props.append(((a,b,cur_size),crop))
        cur_size *= 2  

    print(len(props))
    return props

if __name__ == "__main__":
	run()