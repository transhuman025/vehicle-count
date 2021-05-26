import cv2

cap = cv2.VideoCapture("video.mp4")
vehicle_count=0
#object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=40)

while True:

    ret,frame = cap.read()
    height, width, _ = frame.shape
    print(height,width)
    #extract region of interest
    roi = frame[100:600,900:1800]

    #object detection
    mask = object_detector.apply(roi)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        #calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area < 170:
            continue
           # cv2.drawContours(roi,cnt,-1,(0,255,0),2)
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(roi, (x,y), (x+w,y+h), (0,255,0), 3)
        xmid = int((x+(x+w))/2)
        ymid = int((y+(y+w))/2)
        cv2.circle(roi, (xmid,ymid),5,(0,0,255),5)
        if ymid > 435 and ymid < 465:
            vehicle_count += 1
    cv2.line(frame, (520,450), (2000,450), (0,0,255), 2)   #red line
    cv2.line(frame, (520, 465), (2000, 465), (0, 255, 0), 1)  #offset line
    cv2.line(frame, (520, 435), (2000, 435), (0, 255, 0), 1)   #offset line
    cv2.imshow("ROI",roi)
    cv2.imshow("MASK",mask)
    cv2.putText(frame,'Total Vehicles : {}'.format(vehicle_count),(450,50),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
    cv2.imshow("FRAME", frame)
    key = cv2.waitKey(30)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
