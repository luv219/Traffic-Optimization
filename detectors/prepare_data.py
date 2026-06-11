import os
import csv
import urllib3
import requests
import cv2

# Suppress insecure request warnings from urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")

# Sample limits for quick setup
NUM_TRAIN_IMAGES = 15
NUM_VAL_IMAGES = 5

def create_dirs():
    for folder in [
        os.path.join(IMAGES_DIR, "train"),
        os.path.join(IMAGES_DIR, "val"),
        os.path.join(LABELS_DIR, "train"),
        os.path.join(LABELS_DIR, "val"),
        os.path.join(DATA_DIR, "videos")
    ]:
        os.makedirs(folder, exist_ok=True)
    print("Created directory structure under:", DATA_DIR)

def download_file(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    try:
        response = requests.get(url, verify=False, timeout=15, stream=True)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download successful.")
            return True
        else:
            print(f"Failed to download. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_coco_image(file_name, output_path):
    # Try train2017 first
    url = f"https://images.cocodataset.org/train2017/{file_name}"
    if download_file(url, output_path):
        return True
    # If 404/failed, try val2017
    url = f"https://images.cocodataset.org/val2017/{file_name}"
    return download_file(url, output_path)

def process_dataset(csv_path, split, num_images):
    print(f"\nProcessing {split} dataset from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    # Read annotations
    annotations_by_image = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row["file_name"]
            if file_name not in annotations_by_image:
                annotations_by_image[file_name] = []
            annotations_by_image[file_name].append(row)

    unique_files = list(annotations_by_image.keys())[:num_images]
    print(f"Found {len(annotations_by_image)} unique images. Downloading first {len(unique_files)}...")

    for file_name in unique_files:
        img_dest = os.path.join(IMAGES_DIR, split, file_name)
        
        # Download image if not already present
        if not os.path.exists(img_dest):
            success = download_coco_image(file_name, img_dest)
            if not success:
                print(f"Skipping annotations for {file_name} due to download failure.")
                continue
        
        # Read image to get width and height
        img = cv2.imread(img_dest)
        if img is None:
            print(f"Failed to read image {img_dest}")
            continue
        h, w, _ = img.shape

        # Create YOLO label file
        label_file_name = os.path.splitext(file_name)[0] + ".txt"
        label_dest = os.path.join(LABELS_DIR, split, label_file_name)
        
        with open(label_dest, "w") as lf:
            for ann in annotations_by_image[file_name]:
                # YOLO class indices: coco_91_id - 1
                try:
                    category_id = int(ann["category_id"])
                    yolo_class = category_id - 1
                    
                    x = float(ann["x"])
                    y = float(ann["y"])
                    width = float(ann["width"])
                    height = float(ann["height"])
                    
                    # Convert to normalized center_x, center_y, w, h
                    x_center = (x + width / 2.0) / w
                    y_center = (y + height / 2.0) / h
                    norm_w = width / w
                    norm_h = height / h
                    
                    # Clip coordinates to [0, 1] range to avoid YOLO warnings
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    norm_w = max(0.0, min(1.0, norm_w))
                    norm_h = max(0.0, min(1.0, norm_h))
                    
                    lf.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                except Exception as e:
                    print(f"Error processing annotation row: {e}")

    print(f"Finished processing {split} annotations.")

def create_data_yaml():
    yaml_content = f"""path: {DATA_DIR.replace('\\', '/')}
train: images/train
val: images/val

names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
"""
    yaml_path = os.path.join(ROOT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print("Created data.yaml at:", yaml_path)

def download_sample_video():
    video_url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4"
    video_dest = os.path.join(DATA_DIR, "videos", "traffic.mp4")
    if not os.path.exists(video_dest):
        download_file(video_url, video_dest)
    else:
        print("Sample video already exists.")

if __name__ == "__main__":
    create_dirs()
    process_dataset(os.path.join(ROOT_DIR, "annotation", "train_dataset.csv"), "train", NUM_TRAIN_IMAGES)
    process_dataset(os.path.join(ROOT_DIR, "annotation", "val_dataset.csv"), "val", NUM_VAL_IMAGES)
    create_data_yaml()
    download_sample_video()
    print("Dataset preparation complete!")
