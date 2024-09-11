from ultralytics import YOLO
import cvzone
import cv2
import math
import rospy
from std_msgs.msg import String
from std_msgs.msg import Int32

rospy.init_node("fire_detection_node")
pub_x = rospy.Publisher('fire_detection_x', Int32, queue_size=10)
pub_y = rospy.Publisher('fire_detection_y', Int32, queue_size=10)
rate = rospy.Rate(20)   #10Hz

# Running real time from webcam
cap = cv2.VideoCapture(0)
model = YOLO('/home/huy/Downloads/Telegram Desktop/fire.pt')


# Reading the classes
classnames = ['fire']
while not rospy.is_shutdown():
    ret,frame = cap.read()
    frame = cv2.resize(frame,(720,480))
    result = model(frame,stream=True)
    x0=0
    y0=0
    # Getting bbox,confidence and class names informations to work with
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 60:
                x1,y1,x2,y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),5)
                x0 = (x1+x2)//2
                y0 = (y1+y2)//2

                cv2.circle(frame,(x0,y0),2,(0,0,255),5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5,thickness=2)
                #show stats
                cv2.putText(frame, "x: " + str(x0) + ", y: " + str(y0), (20, 20), cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0, 0, 255), 1, cv2.LINE_AA) 
        pub_x.publish(x0)
        pub_y.publish(y0)
        print("x0="+ str(x0), end="\n\n")
        print("y0="+ str(y0), end="\n\n") 
        x0=0
        y0=0
            
    rate.sleep()


    cv2.imshow('frame',frame)
    cv2.waitKey(1)
