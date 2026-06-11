import os
import cv2
import yaml
import numpy as np
from ultralytics import YOLO

class VehicleCounter:
    def __init__(self, model_path="models/yolo11n.pt", zones_config_path="counting/zones.yaml"):
        # Load YOLO model
        if not os.path.exists(model_path):
            model_path = "yolo11n.pt"  # Fallback to downloading or cached model
        self.model = YOLO(model_path)
        
        # Load zones
        self.zones = []
        if os.path.exists(zones_config_path):
            with open(zones_config_path, "r") as f:
                config = yaml.safe_load(f)
                if config and "zones" in config:
                    for z in config["zones"]:
                        # Convert polygon list of lists to numpy array of shape (N, 1, 2)
                        poly_pts = np.array(z["polygon"], dtype=np.int32)
                        self.zones.append({
                            "name": z["name"],
                            "color": tuple(z["color"]),
                            "polygon": poly_pts,
                            "current_count": 0,
                            "cumulative_ids": set()
                        })
        print(f"Loaded {len(self.zones)} counting zones.")
        
        # Mapping of YOLO COCO class IDs to human names
        # Standard COCO vehicle classes: 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.vehicle_classes = {1, 2, 3, 5, 7}
        self.current_counts = {zone["name"]: 0 for zone in self.zones}
        self.cumulative_counts = {zone["name"]: 0 for zone in self.zones}

    def process_frame(self, frame):
        """Processes a single frame: runs tracking, updates zone counts, draws annotations."""
        # Run YOLO tracking on frame
        # persist=True maintains track IDs across frames
        results = self.model.track(source=frame, persist=True, verbose=False, conf=0.15)
        
        # Reset current frame counts in zones
        for zone in self.zones:
            zone["current_count"] = 0
            
        annotated_frame = frame.copy()
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                xyxys = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.cpu().numpy().astype(int)
                class_ids = boxes.cls.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()
                
                for xyxy, track_id, cls, conf in zip(xyxys, track_ids, class_ids, confidences):
                    # Check if object is a vehicle
                    if cls not in self.vehicle_classes:
                        continue
                    
                    x1, y1, x2, y2 = xyxy
                    # Bottom-center of the bounding box is a reliable point representing vehicle position on road
                    bottom_center = (int((x1 + x2) / 2), int(y2))
                    
                    # Determine which zone the vehicle belongs to
                    in_any_zone = False
                    for zone in self.zones:
                        # cv2.pointPolygonTest returns positive if inside, 0 on boundary, negative if outside
                        is_inside = cv2.pointPolygonTest(zone["polygon"], bottom_center, False) >= 0
                        if is_inside:
                            zone["current_count"] += 1
                            zone["cumulative_ids"].add(track_id)
                            in_any_zone = True
                            
                            # Draw bounding box with zone-specific color
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), zone["color"], 2)
                            label = f"ID:{track_id} {self.model.names[cls]} ({conf:.2f})"
                            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone["color"], 2)
                            break
                    
                    if not in_any_zone:
                        # Draw default white box for vehicles outside defined zones
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
        
        # Update class variables
        for zone in self.zones:
            self.current_counts[zone["name"]] = zone["current_count"]
            self.cumulative_counts[zone["name"]] = len(zone["cumulative_ids"])
            
            # Draw zone polygon on frame
            cv2.polylines(annotated_frame, [zone["polygon"]], True, zone["color"], 2)
            
            # Draw overlay counts near the top/first vertex of the polygon
            first_vertex = zone["polygon"][0]
            label_pos = (first_vertex[0], first_vertex[1] - 10)
            text = f"{zone['name'].upper()}: Active={zone['current_count']} | Total={len(zone['cumulative_ids'])}"
            cv2.putText(annotated_frame, text, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone["color"], 2)
            
        return annotated_frame

    def get_counts(self):
        return {
            "current": self.current_counts.copy(),
            "cumulative": self.cumulative_counts.copy()
        }

    def generate_video_stream(self, video_path):
        """Generator that yields JPEG frames of the processed video for streaming."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return
            
        while True:
            success, frame = cap.read()
            if not success:
                # Loop video when it ends
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            processed_frame = self.process_frame(frame)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        cap.release()

if __name__ == "__main__":
    # Test execution
    video_file = "data/videos/traffic.mp4"
    if os.path.exists(video_file):
        print("Testing VehicleCounter on:", video_file)
        counter = VehicleCounter()
        cap = cv2.VideoCapture(video_file)
        success, frame = cap.read()
        if success:
            out = counter.process_frame(frame)
            cv2.imwrite("runs/detect/infer/test_counter_frame.jpg", out)
            print("Successfully processed a test frame and saved to runs/detect/infer/test_counter_frame.jpg")
            print("Current Counts:", counter.get_counts())
        cap.release()
    else:
        print("Test video not found. Please prepare the dataset first.")
