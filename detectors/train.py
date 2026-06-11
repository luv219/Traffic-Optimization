import argparse
import os
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 model on custom Traffic dataset")
    parser.add_argument("--model", type=str, default="models/yolo11n.pt", help="Path to base model weights")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml dataset config")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on (e.g., cpu, 0, cuda)")
    args = parser.parse_args()

    # Verify model file exists
    if not os.path.exists(args.model):
        print(f"Model path {args.model} does not exist. Using yolo11n.pt.")
        args.model = "yolo11n.pt"

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"Starting training on data={args.data} for {args.epochs} epochs...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project="runs/train",
        name="traffic_yolo",
        workers=0  # Use 0 workers for compatibility/debugging on Windows
    )
    print("Training finished! Results saved in runs/train/traffic_yolo")

if __name__ == "__main__":
    main()
