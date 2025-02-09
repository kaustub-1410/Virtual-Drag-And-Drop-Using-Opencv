import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

# Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Initialize Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)  # Detect up to 1 hand
colorR = (255, 0, 255)  # Rectangle color

# Drag Rectangle Class
class DragRect:
    def __init__(self, posCenter, size=(200, 200)):
        self.posCenter = list(posCenter)
        self.size = size
        self.dragging = False  # Added dragging state

    def update(self, cursor, fingers):
        cx, cy = self.posCenter
        w, h = self.size

        # If index finger is inside rectangle and middle finger is close, start dragging
        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            if fingers[1] == 1 and fingers[2] == 0:  # Index up, middle down (grabbing)
                self.dragging = True

        # Update position if dragging
        if self.dragging:
            self.posCenter = list(cursor)

        # Release dragging if fingers open
        if fingers[1] == 1 and fingers[2] == 1:  # Both index and middle up (release)
            self.dragging = False

# Create Multiple Draggable Rectangles
rectList = [DragRect((x * 250 + 150, 150)) for x in range(3)]  # 3 Rectangles

while True:
    success, img = cap.read()
    if not success:
        continue
    img = cv2.flip(img, 1)  # Flip for mirror effect

    # Detect Hands
    hands, img = detector.findHands(img, draw=True)  # ✅ Correct function
    if hands:
        for hand in hands:
            lmList = hand["lmList"]  # ✅ Get landmarks
            fingers = detector.fingersUp(hand)  # ✅ Detect which fingers are up

            if len(lmList) >= 12:  # Ensure valid hand detection
                cursor = lmList[8][:2]  # Index finger tip position

                for rect in rectList:
                    rect.update(cursor, fingers)

    # Transparent Layer for Rectangles
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    # Overlay Transparent Layer
    out = cv2.addWeighted(img, 0.5, imgNew, 0.5, 0)

    # Display the Output
    cv2.imshow("Virtual Drag & Drop", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
