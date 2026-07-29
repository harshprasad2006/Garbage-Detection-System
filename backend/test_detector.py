from detector import GarbageDetector

# Load the trained model
detector = GarbageDetector(model_path="../models/best.pt")

# Replace this with an actual test image filename
image_path = r"C:\Users\harsh\OneDrive\Desktop\HARSH\Internship_Proj\BeRamround2\Garbage-Detection-System\dataset\test\images\battery_91_jpg.rf.4ae8e3736ac8895c682d3773e7f35e0a.jpg"
annotated_image, detections = detector.detect(image_path)

print("\nDetections Found:\n")

for detection in detections:
    print(detection)