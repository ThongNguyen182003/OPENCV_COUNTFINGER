import cv2
import mediapipe as mp
import time
import math
import os
import firebase_admin
from firebase_admin import credentials, db
import threading

# Firebase
path_to_json = "C:/Users/thong/OneDrive/Máy tính/finger/source/firebase_admin-key.json"

cred_obj = firebase_admin.credentials.Certificate(path_to_json)
default_app = firebase_admin.initialize_app(cred_obj, {
    'databaseURL': 'https://opencv-numberfinger-default-rtdb.europe-west1.firebasedatabase.app/'
})

def distance(x1,y1,x2,y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

class handDetector():
    def __init__(self, mode = False, maxHands = 2,modelComplexity=1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self,img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks: # draw landmarks
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw = True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return lmlist
def getNumber(ar):
    count = 0 
    for id in ar :
        if id == 1 :
            count = count + 1
    return (count)

# def sendData(img,ref, numberFingers ):
#     try:    
#         ref.set(numberFingers)

#     except Exception as e:
#         print("Error:", e)
        
def main():
    ref = db.reference("fingersNumber")
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    countdown_timer = 5 
    countdown_started = False

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        tipId=[4,8,12,16,20]
        
        if len(lmlist) != 0:  # detect have finger
            finger = []
            if(lmlist[tipId[0]][1]>lmlist[tipId[0]-1][1]):
                finger.append(0)
            else :
                finger.append(1)
            for id in range(1,len(tipId)) :
                if(lmlist[tipId[id]][2]<lmlist[tipId[id]-3][2]):
                    finger.append(1)
                else :
                    finger.append(0)
            print(finger)
            numberFingers = getNumber(finger)
            cv2.putText(img,str(numberFingers),(45,375),cv2.FONT_HERSHEY_PLAIN, 5,(255,0,0),5)
            try:    
                ref.set(numberFingers)
            except Exception as e:
                print("Error:", e)
        else : 
            cv2.putText(img,"NO HAND",(45,375),cv2.FONT_HERSHEY_PLAIN, 5,(255,0,0),5)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if  cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()