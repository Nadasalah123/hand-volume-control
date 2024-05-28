import cv2
import mediapipe as mp

import numpy as np
import math

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER

cap=cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands=mpHands.Hands(min_detection_confidence=0.7)
mpDraw=mp.solutions.drawing_utils

devices=AudioUtilities.GetSpeakers()
interface=devices.Activate(
    IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume=cast(interface,POINTER(IAudioEndpointVolume))
volRange=volume.GetVolumeRange()

minVol=volRange[0]
maxVol=volRange[1]
print(volRange)

while True:
    success,img=cap.read()
    img =cv2.flip(img,1)
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    lmList=[]
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])

                mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
                
                if len(lmList)==21:
                    x1, y1 = lmList[4][1], lmList[4][2]
                    x2, y2 = lmList[8][1], lmList[8][2]
                    cx, cy =(x1 + x2) // 2, (y1 +y2) // 2

                    cv2.circle(img,(x1,y1),8,(255,0,255),cv2.FILLED)
                    cv2.circle(img,(x2,y2),8,(255,0,255),cv2.FILLED)
                    cv2.line(img, (x1,y1),(x2,y2),(0,0,0),3)
                    cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
                    
                    length=math.hypot(x2-x1,y2-y1)
                    #print(length)
                    #lengh=50-200

                    if length<50:
                        cv2.circle(img,(cx,cy),5,(0,0,255),cv2.FILLED)
                    if length>200:
                        cv2.circle(img,(cx,cy),5,(0,255,0),cv2.FILLED)
                    
                    vol=np.interp(length,[50,200],[minVol,maxVol])
                    volume.SetMasterVolumeLevel(vol,None)
    cv2.imshow("Image",img)
    if cv2.waitKey(5) & 0xFF==27:
        break