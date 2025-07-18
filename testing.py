from ultralytics import YOLO

model = YOLO("traffic_light_detection/lisa_finetune_v14/weights/best.pt")

results = model.predict("test_image_1.png")

results[0].show()
results[0].save("test_image_1_results.png")