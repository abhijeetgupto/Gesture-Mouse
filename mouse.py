import numpy as np
import HandDetector as htm
import time
import pyautogui
import cv2

pyautogui.FAILSAFE = False
##########################
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 3
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector()
wScr, hScr = pyautogui.size()
# print(wScr, hScr)

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)

    # 3. Check which fingers are up
    fingers = detector.fingersUp()
    # print(fingers)
    img = cv2.rectangle(img, (100, 50), (540, 250), (0, 0, 255), thickness=1)

    # 4. Only Index Finger : Moving Mode
    if fingers == [0, 1, 0, 0, 0]:
        # 5. Convert Coordinates of the rectangle to that of the screen
        x3 = np.interp(x1, (100, wCam - 100), (0, wScr))
        y3 = np.interp(y1, (50, hCam - 250), (0, hScr))

        # 6. Smoothen Values
        clocX = (plocX + (x3 - plocX) / smoothening)
        clocY = (plocY + (y3 - plocY) / smoothening)

        # 7. Move Mouse
        pyautogui.moveTo(clocX, clocY)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY

    # 8. Both Index and middle fingers are up : Clicking Mode
    elif fingers == [0, 1, 1, 0, 0]:
        # 9. Find distance between fingers
        length, img, lineInfo = detector.findDistance(8, 12, img)
        # print(length)
        # 10. Click mouse if distance short
        if length < 40:
            cv2.circle(img, (lineInfo[4], lineInfo[5]),
                       15, (0, 255, 0), cv2.FILLED)
            pyautogui.click()

    elif fingers == [0, 0, 1, 0, 0]:
        cv2.destroyAllWindows()


    # 11. Frame Rate (FPS Calculation)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    # 12. Displaying the final image
    cv2.imshow("Image", img)
    cv2.waitKey(1)

    #12. Closing the window after pressing "X" on window.
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()