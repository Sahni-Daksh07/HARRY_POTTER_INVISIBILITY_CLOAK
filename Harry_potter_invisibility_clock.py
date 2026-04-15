import cv2
import numpy as np
import time

### PARAMETERS

Capture_BG_Seconds = 3
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

''' HSV color range for RED cloak '''

lower_red_1 = np.array([0,120,70])
upper_red_1 = np.array([10,255,255])
lower_red_2 = np.array([170,120,70])
upper_red_2 = np.array([180,255,255])

''' If anyone is using greeen cloak, then uncomment the code given below and comment the code given above which is for the RED cloak ''' 

# lower_green = np.array([40,50,50])
# upper_green = ap.array([90,255,255])

''' Camera setup '''

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)


time.sleep(1)
print(f"Capturing background for {Capture_BG_Seconds} seconds... Please move out of the frame.")

# Capture background frames

bg_frames = []
start = time.time()
while time.time() - start < Capture_BG_Seconds:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    bg_frames.append(frame)


# Capture Mediam Background

background = np.median(bg_frames, axis=0).astype(np.uint8)
print("Background captures! Now wear your cloak. Press 'q' to quit.")

kernel = np.ones((3,3), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mark for red

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = mask1 + mask2

    # If using green:
    # mask = cv2.inRAnge(hsv, lower_green, upper_green)

    # Clean Mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    mask_inv = cv2.bitwise_not(mask)

    # Replace cloak area with background

    part1 = cv2.bitwise_and(background, background, mask=mask)
    part2 = cv2.bitwise_and(frame, frame, mask=mask_inv)
    final = cv2.addWeighted(part1, 1, part2, 1, 0)

    cv2.imshow("Invisibility Cloak", final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

