from ultralytics import YOLO

model = YOLO("traffic_light_detection/lisa_finetune_v1/weights/best.pt")

results = model.predict("test_image_1.webp")

results[0].show()
results[0].save("test_image_2_results.jpg")