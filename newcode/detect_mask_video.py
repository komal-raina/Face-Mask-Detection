# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import smtplib
from email.message import EmailMessage
import imghdr



email_subject = "Komal Email Face Mask"
sender_email_address = "komal.sharma19@vit.edu"
receiver_email_address = "komalrainajammu@gmail.com"
email_smtp = "smtp.gmail.com"
email_password = "phkmsnlqcobejvyt"

message = EmailMessage()

message['Subject'] = email_subject
message['From'] = sender_email_address
message['To'] = receiver_email_address


def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	
	faces = []
	locs = []
	preds = []

	
	for i in range(0, detections.shape[2]):
		
		confidence = detections[0, 0, i, 2]


		if confidence > 0.5:
			
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

		
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
	
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	
	return (locs, preds)

prototxtPath = r"C:\Users\91600\vsfolders\python\Face mask detection\Code\face_detector\deploy.prototxt"
weightsPath = r"C:\Users\91600\vsfolders\python\Face mask detection\Code\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model("C:\\Users\\91600\\vsfolders\\python\\Face mask detection\\Code\\mask_detector.model")

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

glabel = ""
counter = 0
intrudirectory = r"C:\Users\91600\vsfolders\python\Face mask detection\Code\Intruders"
accesseddirectory = r"C:\Users\91600\vsfolders\python\Face mask detection\Code\Accessed"



while True:
	
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	
	for (box, pred) in zip(locs, preds):
		
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		
		if glabel == label:
			counter += 1
		else:
			counter = 0
			glabel = label

		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	

	if counter >= 50:
		if glabel == "No Mask":
			os.chdir(intrudirectory)
			cv2.imwrite("intruder.jpg", frame)
			with open('intruder.jpg', 'rb') as file:
				image_data = file.read()
			message.set_content("An intruder was found")

		elif glabel == "Mask":
			os.chdir(accesseddirectory)
			cv2.imwrite("accesses.jpg", frame)
			with open('accesses.jpg', 'rb') as file:
				image_data = file.read()
			message.set_content("A person has been granted access")
		
		message.add_attachment(image_data, maintype='image', subtype=imghdr.what(None, image_data))

		server = smtplib.SMTP(email_smtp, '587')
		server.ehlo()
		server.starttls()
		server.login(sender_email_address, email_password)
		server.send_message(message)
		server.quit()

		
		vs.stream.release()
		vs.stop()
		cv2.destroyAllWindows()
		break
	

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stream.release()
vs.stop()