#This program uses cv2 module that detects when someone smiles :)
#Created by Callyn Villanueva (updated Dec3)

import cv2

#xml files uploaded
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


vidcappy = cv2.VideoCapture(0)
#error checking purposes
if not vidcappy.isOpened():
        print('camera not opened! please check if properly imported cv2')


while True:
    #Ret will obtain return value from getting the camera frame, either true of false (boolean).
    ret, image = vidcappy.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    '''
    the detectMultiScale() method finds rectangular regions in the given image that are likely to contain objects
    the cascade has been trained for and returns those regions as a sequence of rectangles'''

    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    smile = smile_cascade.detectMultiScale(gray, 1.8, 20)

    #nested for loop creating rectangle for face
    for (fx,fy,fw,fh) in face:
        cv2.rectangle(image,(fx,fy),(fx+fw,fy+fh),(300,300,0),2)
        # creating rectangle for smile
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(image, (sx, sy), ((sx + sw), (sy + sh)), (300,300,0), 2)


    cv2.imshow('img',image)

    #saving image to desktop! :)



    #keyboard binding function! Press escape to exit the program!
    escape_key = cv2.waitKey(30)

    #asking user of they want a copy of the image
    if escape_key == 27:
        userInput = input("Would you like your picture take? Type yes or no on the console")
        if(userInput == "yes"):
            cv2.imwrite("outputImage.jpg", image)
            print("Finished Image Recognition - Image was taken!")
            break
        elif(userInput == "no"):
            break

vidcappy.release()
cv2.destroyAllWindows()