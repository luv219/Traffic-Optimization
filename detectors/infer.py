import argparse
import os
import cv2
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv11 inference on traffic images/videos")
    parser.add_argument("--model", type=str, default="models/yolo11n.pt", help="Path to model weights")
    parser.add_argument("--source", type=str, default="data/images/train/000000140006.jpg", help="Path to image, video or folder")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save-dir", type=str, default="runs/detect/infer", help="Directory to save visualized outputs")
    args = parser.parse_args()

    # Verify model and source
    if not os.path.exists(args.model):
        print(f"Model path {args.model} does not exist. Using yolo11n.pt.")
        args.model = "yolo11n.pt"
    
    if not os.path.exists(args.source):
        print(f"Source path {args.source} does not exist. Please specify a valid file.")
        return

    # Load YOLO model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Make output directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Run inference
    print(f"Running inference on: {args.source}")
    results = model(args.source, conf=args.conf)

    # Process results
    for i, r in enumerate(results):
        # Original image
        orig_img = r.orig_img.copy()
        
        # Plot detections on image
        # plot() returns a numpy array with annotations drawn
        annotated_frame = r.plot()
        
        # Save output
        source_name = os.path.basename(args.source)
        if len(results) > 1:
            dest_name = f"result_{i}_{source_name}"
        else:
            dest_name = f"result_{source_name}"
            
        output_path = os.path.join(args.save_dir, dest_name)
        cv2.imwrite(output_path, annotated_frame)
        print(f"Saved inference results to: {output_path}")

if __name__ == "__main__":
    main()
