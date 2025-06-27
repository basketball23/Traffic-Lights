from ultralytics import YOLO

model = YOLO("traffic_light_detection/lisa_finetune_v13/weights/best.pt")

results = model.predict("test_image_2.jpg")

results[0].show()
results[0].save("test_image_2_results.jpg")