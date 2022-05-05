import numpy as np
import cv2 
from imutils.contours import sort_contours
import imutils
from tensorflow.keras.models import load_model

model2 = load_model('model.h5')
symbols = ['+','1', '2', '3', '4', '5', '6', '7', '8', '9','=','X','^','y']

class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret,frame=self.video.read()

        #faces=faceDetect.detectMultiScale(frame, 1.3, 5)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, im_th= cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
        cnts = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sort_contours(cnts, method="left-to-right")[0]
        chars=[]
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if w*h>1200:
                roi = gray[y:y + h, x:x + w]
                img = cv2.resize(roi,(45, 45))
                norm_image = cv2.normalize(img, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                norm_image = norm_image.reshape((norm_image.shape[0], norm_image.shape[1], 1))
                case = np.asarray([norm_image])
                pred = model2.predict([case])
                chars.append(symbols[np.argmax(pred)])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, str(symbols[np.argmax(pred)]), (x, y), cv2.FONT_HERSHEY_DUPLEX,  0.7, (0,0,255), 2) 
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()