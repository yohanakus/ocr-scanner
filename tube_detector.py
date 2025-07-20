from ultralytics import YOLO

# Load your trained model from Roboflow
model = YOLO("runs/detect/train/weights/best.pt")  # Change to your correct model path if needed

# Path to the image you want to analyze
image_path = "screenshot.jpg"

# Run inference
results = model(image_path)

# Get detections (bounding boxes)
boxes = results[0].boxes

# Count the number of detected tubes
tube_count = len(boxes)

# Output the result
print(f"🧪 Detected {tube_count} tube(s) in the image.")
