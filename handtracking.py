"""Using mediapipe to create hand tracking project which includes palm detection and hand landmarks.
-palm detection provides a cropped image of the hand
-hand landmarks finds 21 different landmarks
This project also includes the face detection"""
import cv2 #importing Open CV
import mediapipe as mp #importing mediapipe
import time #to check the frame rate

cap = cv2.VideoCapture(0) #creating video object using webcam number 0
hands_object = mp.solutions.hands
hands = hands_object.Hands()#creating an object of Hands
mp_draw = mp.solutions.drawing_utils#to draw the 21 points on the hand

prev_time = 0
current_time = 0

while True:
    success, img = cap.read()#to open the webcam
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#converting the image to RGB before passing it into the hands object
    results = hands.process(imgRGB)#process the image (frames) and return the hand landmarks
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handslmks in results.multi_hand_landmarks:
            for id, lm in enumerate(handslmks.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            mp_draw.draw_landmarks(img, handslmks, hands_object.HAND_CONNECTIONS)#drawing landmarks connections
    current_time = time.time()
    fps = 1/(current_time-prev_time)
    prev_time = current_time

    cv2.putText(img, str(int(fps)), (10,78), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255), 3)#printing fps on the picture
    cv2.imshow("Image", img)#showing the webcam
    cv2.waitKey(1)