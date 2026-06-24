import cv2
from ultralytics import YOLO

model = YOLO("models/yolo11n.pt")
cap = cv2.VideoCapture("data/videos/traffic.mp4")

# Let's inspect frame 100
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
if ret:
    print(f"Frame 100 shape: {frame.shape}")
    # Run standard inference with low confidence
    res = model(frame, conf=0.01)
    boxes = res[0].boxes
    if boxes is not None and len(boxes) > 0:
        print(f"Detected {len(boxes)} objects at conf=0.01:")
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            print(f"  Class {cls} ({model.names[cls]}): conf {conf:.4f}, box {xyxy}")
    else:
        print("No detections even at conf=0.01!")
else:
    print("Could not read frame 100")
cap.release()
