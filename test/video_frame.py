import numpy as np
import cv2

def run():
	cap = cv2.VideoCapture(0)



	while(True):
	    # Capture frame-by-frame
	    ret, frame = cap.read()

	    # Our operations on the frame come here
	    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY
	    getRectangle(frame)

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


if __name__ == "__main__":
	run()