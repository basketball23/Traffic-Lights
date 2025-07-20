from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict("test.jpg")

results[0].show()
results[0].save("test-image-2-results.jpg")