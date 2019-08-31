#This program uses cv2 module that detects when someone smiles :)
#Callyn Villanueva

import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
cap = cv2.VideoCapture(0)



while True:
    #Ret will obtain return value from getting the camera frame, either true of false.
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    smile = smile_cascade.detectMultiScale(gray, 1.8, 20)

    #creating rectangle for face
    for (fx,fy,fw,fh) in face:
        cv2.rectangle(image,(fx,fy),(fx+fw,fy+fh),(300,300,0),2)

        # creating rectangle for smile
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(image, (sx, sy), ((sx + sw), (sy + sh)), (300,300,0), 2)

    cv2.imshow('img',image)

    #keyboard binding function! Press escape to exit the program!
    k = cv2.waitKey(30) & 0xff


    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()