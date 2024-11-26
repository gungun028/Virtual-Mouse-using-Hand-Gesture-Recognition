'''
import cv2
import autopy
import time
import numpy as np
import pyHandTrackingModule as htm

whCam,hhCam = 640, 480
frameR = 100
smoothening = 7

pTime = 0
plocx,plocy = 0, 0
clocx,clocy = 0, 0
cap = cv2.VideoCapture(0)
cap.set(3,whCam)
cap.set(4,hhCam)

detector = htm.handDetector(maxHands=1)

wScr, hScr = autopy.screen.size()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingerUp()
        cv2.rectangle(img,(frameR,frameR), (whCam - frameR, hhCam - frameR),
                      (255, 0, 255), 2)
        if fingers[1] == 1 and fingers[2] == 0:
            x2, y2 = lmList[12][1:]

            x3 = np.interp(x1, (frameR, whCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hhCam - frameR), (0, hScr))
            # 6. Smoothen Values
            clocX = plocx + (x3 - plocx) / smoothening
            clocY = plocy + (y3 - plocy) / smoothening

            autopy.mouse.move(wScr - clocx, clocy)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocx, plocy = clocx, clocy

        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            # 10. Click mouse if distance short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
'''
import cv2
import mediapipe as mp
import pyautogui


x1 = y1 = x2 = y2 = 0
webcam = cv2.VideoCapture(0)
my_hands = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils


while True:
    _, image = webcam.read()
    image = cv2.flip(image,1)
    frame_height, frame_width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = my_hands.process(rgb_image)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(image, hand)
            landmarks = hand.landmark

            for id,landmark in enumerate(landmarks):
                x = int(landmark.x*frame_width)
                y = int(landmark.y*frame_height)

                if id == 8:  #index finger
                    cv2.circle(img = image, center = (x,y),radius = 8, color = (0,255,255),thickness= 3)
                    x1 = x
                    y1 = y
                if id == 4: #thumb
                    cv2.circle(img = image, center = (x,y) ,radius = 8, color = (0,0,255),thickness= 3)
                    x2 = x
                    y2 = y

                if id == 12: # middle finger
                    cv2.circle(img=image, center=(x, y), radius=8, color=(0, 0, 255), thickness=3)
                    x3 = x
                    y3 = y

        volumecontrol = ((x2-x1)**2 + (y2-y1)**2)**0.5//4
        ''''
        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),5)
        if volumecontrol>25:
            pyautogui.press("volumeup")
        else:
            pyautogui.press("volumedown")
        '''

        doubleclickcontrol = ((x3 - x2) ** 2 + (y3 - y2) ** 2) ** 0.5
        if doubleclickcontrol < 20:
            pyautogui.doubleClick()

        #print(y3-y1)
        if y2 < y1 - 10:
            pyautogui.scroll(1)  # Scroll up
        # Check for scrolling down
        elif y2 > y1 + 10:
             pyautogui.scroll(-1)  # Scroll down








    cv2.imshow("hand volume using python ",image)

    key = cv2.waitKey(10)
    if key == 27:
        break
webcam.release()
cv2.destroyAllWindows()















