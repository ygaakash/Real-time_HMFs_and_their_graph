#!/usr/bin/env python
# coding: utf-8

# # REAL-TIME TEXT DETECTION USING OPENCV AND PYTESSERACT.

# In[ ]:


#before executing the code make sure you first setup pytesseract on your computer. Also install it through anaconda prompt as 
# pip install pytesseract


# In[ ]:


#importing pytesseract
import pytesseract


# In[ ]:


#importing opencv
import cv2 
#install opencv through prompt as 
#pip install opencv-python if this line of code shows error. 


# In[ ]:


#configuring pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
#error on this line is probably because the location of your tessseract.exe is different than mine.


# In[ ]:


front_scale = 1.5
font= cv2.FONT_HERSHEY_PLAIN


# In[ ]:


#loading the webcam
cap= cv2.VideoCapture(1)


# In[ ]:


#if webcam couldnt be read for what so reasons.
if not cap.isOpened():
    cap= cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot Open Video")


# In[ ]:


#please watch my video explanation to understand the below codes. 
cntr =0;
while True:
    ret, frame =cap.read()
    cntr= cntr+1;
    if ((cntr%20)==0):
        
            imgH, imgW,_ = frame.shape
            
            x1,y1,w1,h1 = 0,0,imgH,imgW
            
            imgchar =pytesseract.image_to_string(frame)
            
            imgboxes= pytesseract.image_to_boxes(frame)
            for boxes in imgboxes.splitlines():
                boxes= boxes.split(' ')
                x,y,w,h= int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])
                cv2.rectangle(frame, (x, imgH-y), (w, imgH-h),(0,0,255),3)
                
                cv2.putText(frame, imgchar, (x1 + int(w1/50),y1 + int(h1/50)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)
                
                font= cv2.FONT_HERSHEY_SIMPLEX
                
                cv2.imshow('Text detection',frame)
                
                if cv2.waitKey(2) & 0xFF == ord('a'):
                    break
                    
cap.release()
cv2.destroyAllWindows()


# In[ ]:




