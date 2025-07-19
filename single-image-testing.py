from ultralytics import YOLO

model = YOLO("training/traffic-light-detection/lisa-finetune-v1/weights/best.pt")

results = model.predict("test-image-1.webp")

results[0].show()
results[0].save("test-image-1-results.jpg")