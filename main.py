import cv2
from datetime import datetime
import os
import time

cv2.namedWindow("Security")
security_camera = cv2.VideoCapture(0)  # get webcam
security_camera.set(cv2.CAP_PROP_FPS, 30)  # set FPS to 30
mog = cv2.createBackgroundSubtractorMOG2()  # set up background subtraction
print(security_camera.get(cv2.CAP_PROP_FRAME_WIDTH), security_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
if security_camera.isOpened():  # try to get the first frame
    ret, frame = security_camera.read()
else:
    ret = False

while ret:
    ret, frame = security_camera.read()  # Read current frame
    cv2.imshow("Security", frame)  # Display frame

    # Clean up background subtraction for motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    foreground_mask = mog.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foreground_mask = cv2.erode(foreground_mask, kernel, iterations=1)
    foreground_mask = cv2.dilate(foreground_mask, kernel, iterations=1)
    # cv2.imshow("Security", foreground_mask)

    contours, hierarchy = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Write current frame as image if movement is detected
    for contour in contours:
        if cv2.contourArea(contour) < 15000:
            continue
        # Create a folder for a new day if one does not exist
        path = "Y" + str(datetime.now().year) + "_M" + str(datetime.now().month) + "_D" + str(datetime.now().day)
        print(path)
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)

        # This solves two problems:
        # 1. Prevents constant overwrites to existing jpg image and thus prevents disk wear.
        # 2. We do not care about who is coming into the room, We will only get their back.
        # (This assumes mounting the camera above the door instead being in plain sight as one enters the room)
        # Capturing the first frame in the second will gain a greater chance to capture the face
        # of the person LEAVING the room, instead of just a clip of their head if they are out of frame
        # near the end of the second in an end of second frame capture setup.
        # If you prefer end of second frame capture, remove the next few lines containing "isExist".
        path = path + "/" + str(datetime.now().strftime("%HH_%MM_%SS")) + ".jpg"
        isExist = os.path.exists(path)
        if not isExist:
            cv2.imwrite(path, frame)
        # time.sleep(1)

    key = cv2.pollKey()
    if key == 27:  # exit on ESC
        break

security_camera.release()
cv2.destroyWindow("Security")
