import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
# import cv2
import json
import os





def extract(name):
    video = cv2.VideoCapture(name)
    if not video.isOpened():
        print("Error opening video file")
        exit()
    tracking_data = []


    reducer=0
    while True:
        success, frame = video.read()
        if not success:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range of colors to detect (for object tracking)
        lower_color = (0, 0, 0)
        upper_color = (180, 255, 50)

        # Threshold the HSV image to get only the desired colors (for object tracking)
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Load a pre-trained face detection model (for face tracking)
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        
        # Convert the frame to grayscale (for face tracking)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame (for face tracking)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Check if any faces or objects were detected
        if len(faces) == 0:
            print("No faces or objects detected")
            break

        # Select the first face or object
        face = faces[0]

        # Extract the face or object coordinates
        x, y, w, h = face

        # Use the MeanShift algorithm to track the face or object
        track_window = cv2.meanShift(gray[y:y+h, x:x+w], (x, y, w, h), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))

        # Extract the tracking window coordinates
        cx, cy = track_window

        # Draw a rectangle around the tracked face or object
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Store the tracking data
        # if reducer %2==0:
        tracking_data.append((cx, cy))
        # reducer+=1

    # Release the video capture
    video.release()

    # Save the tracking data to a file
    track= [tr[1][:2] for tr in tracking_data ]
    return track


def change(v1,v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dt = 1
    dr = (v2 - v1) / dt
    return dr

def prepare(Clist):
    vector_list=[]
    for i in range(0,len(Clist)):
        vector_list.append(list(change(Clist[i-1],Clist[i])))
    if len(vector_list) <200:
        print("video incompatible")
        return
    return vector_list[0:200]
 
 
 
 
        

data=prepare(extract("attack_highdef_client007_session01_highdef_photo_controlled.mov"))



X=[]
X.append(data[0:200])

X= np.array(X) 
model = tf.keras.models.load_model('my_model.h5')
preds = model.predict(X)
predicted_class = np.argmax(preds, axis=1)

print(predicted_class)