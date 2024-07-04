import cv2, time, mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands()
ptime = 0
ctime = 0

while True:
    success, img = cap.read()
    imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRBG)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for i in results.multi_hand_landmarks:
            for id, lm in enumerate(i.landmark):
                print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)  # for finger numbers
                if id == 4:
                    cv2.circle(img, (cx, cy), 12, (0, 255, 0), cv2.FILLED)
            mpDraw.draw_landmarks(img, i, mpHands.HAND_CONNECTIONS)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (24,69), cv2.FONT_ITALIC, 2, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

