import cv2
import tensorflow as tf
import numpy as np

emotions  = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', "surprise"]

# Load the model
model = tf.keras.models.load_model('emotions/model/emotions.h5')

# frontal face detection classifier
face_cascade = cv2.CascadeClassifier('face/haarcascade_frontalface_default.xml')

vid = cv2.VideoCapture(0)

while(True):
	# Capture the video frame
	# by frame
	ret, frame = vid.read()
	
	# Convert the frame to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect the faces in the frame
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	# Draw rectangle around the faces
	for (x,y,w,h) in faces:
		# Crop the image frame into face section
		roi_face = frame[y:y+h, x:x+w]

		cv2.imshow('Face', roi_face)

		#cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

		# Predicting the emotion
		
		# resize the frame to feed to the emotions model
		INPUT_RES = (150, 150)

		image = tf.image.resize(roi_face, INPUT_RES)
		input_arr = tf.keras.preprocessing.image.img_to_array(image)/255.
		input_arr = np.array(input_arr[np.newaxis, ...])

		# get the predicted emoji
		predictions = model.predict(input_arr)
		prediction = np.argmax(predictions)
		emotion = emotions[prediction]

		emoji = cv2.imread("emotions/emoji/{}.png".format(emotion), -1)
		emoji = cv2.resize(emoji, (w, h))

		# put the image on the frame
		y1, y2 = y, y + h
		x1, x2 = x, x + w

		# make it transparent background
		alpha_s = emoji[:, :, 3] / 255.0
		alpha_l = 1.0 - alpha_s
		
		for c in range(0, 3):
			frame[y1:y2, x1:x2, c] = (alpha_s * emoji[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])

		break
	
	# Display the resulting frame
	cv2.imshow('Emotion filter', frame)
	
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()