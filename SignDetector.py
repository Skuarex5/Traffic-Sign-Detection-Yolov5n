from ultralytics import YOLO
import cv2
import torch
import gc
import imutils

model = YOLO('model.pt')
model.to('cuda')

cap = cv2.VideoCapture(0)

first_frame = None

def detect_motion(curr_frame, ref_frame):
    gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    frame_delta = cv2.absdiff(ref_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) >= 200:
            return True
    return False


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    motion = detect_motion(frame, first_frame)

    if motion:
        with torch.no_grad():
            results = model(frame)
        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame

    cv2.imshow("YOLO Detection (motion-triggered)", annotated_frame)
    
    torch.cuda.empty_cache()
    gc.collect()

    first_frame = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
torch.cuda.empty_cache()
gc.collect()

