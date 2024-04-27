from cvzone.HandTrackingModule import HandDetector
import cv2
import socket

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

success, img = cap.read()
h, w, _ = img.shape
detector = HandDetector(detectionCon=0.8, maxHands=2)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

flip = False

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    data = []
    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        for lm in lmList1:
            data.extend([lm[0], h - lm[1], lm[2]])

        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            for lm in lmList2:
                data.extend([lm[0], h - lm[1], lm[2]])

    sock.sendto(str.encode(str(data)), serverAddressPort)

    if cv2.waitKey(1) == 27:  # 'Esc' key to exit
        break

    if cv2.waitKey(1) == ord('f'):
        flip = not flip

    if flip:
        img = cv2.flip(img, 1)  # Flip horizontally

    cv2.imshow("Image", img)