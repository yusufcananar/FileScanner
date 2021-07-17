import cv2
import numpy as np

#Constants
imHeight = 640
imWidth  = 480
kernel   = np.ones((5,5))

def preProcessing(src):
    imGray      = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    imBlur      = cv2.GaussianBlur(imGray, (5,5),1)
    imCanny     = cv2.Canny(imBlur, 120, 200)
    imDilation  = cv2.dilate(imCanny, kernel, iterations=3)
    imErosion   = cv2.erode(imDilation, kernel, iterations=1)

    return imErosion

def getContours(src):
    maxArea = 0
    biggest = np.array([])

    contours, hierarchy = cv2.findContours(src, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 4000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            cornerCount = len(approx)
            if area > maxArea and cornerCount == 4:
                biggest = approx
                maxArea = area
                cv2.drawContours(img, biggest, -1, (255, 0, 0), 20)

    return biggest

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)

    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints,axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew

def getWarp(src, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [imWidth, 0], [0, imHeight], [imWidth, imHeight]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imRes = cv2.warpPerspective(src, matrix, (imWidth, imHeight))
    imRes = imRes[10:imRes.shape[0]-10, 10:imRes.shape[1]-10]
    imRes = cv2.resize(imRes,(imWidth,imHeight))

    return imRes


# Read and Show video or webcam feed
cap = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)
cap.set(10,150)

while True:
    success, frame = cap.read()
    if success:
        frame = cv2.resize(frame, (640, 480))
        img = frame.copy()

        imThresh = preProcessing(img)
        biggest = getContours(imThresh)
        if len(biggest) == 4:
            imRes = getWarp(frame, biggest)
            cv2.imshow("Result Stream", imRes)

        cv2.imshow("Main Stream", img)
        cv2.imshow("Thresh", imThresh)
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            break
        elif k == ord('c'):
            pass
    else:
        print("Video capture is " + success)
        break