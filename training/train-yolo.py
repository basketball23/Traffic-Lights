import os
from ultralytics import YOLO

# Define paths
DATA_YAML_PATH = './data.yaml' # Path to dataset.yaml file
MODEL_SIZE = 'yolov8n.pt'    # YOLOv8 model size
PROJECT_NAME = 'traffic-light-detection'
EXPERIMENT_NAME = 'lisa-finetune-v1'
EPOCHS = 10                 # Number of training epochs
IMG_SIZE = 640               # Image size
BATCH_SIZE = 32           # Batch size

def train_yolo_model():
    print(f"Starting YOLOv8 training with {MODEL_SIZE} on {DATA_YAML_PATH}")

    # Load a pre-trained YOLOv8 model

    model = YOLO("traffic-light-detection/lisa-finetune-v1/weights/last.pt")  # Load your pre-trained model

    # Ensure the data.yaml exists
    if not os.path.exists(DATA_YAML_PATH):
        print(f"Error: data.yaml not found at {DATA_YAML_PATH}")
        print("Please check the path and ensure your dataset is correctly structured.")
        return

    # Train the model
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        patience=50,  # Stop if no improvement for 50 epochs
        device='mps',
        val=False
    )

    print("Training complete! Results saved to:")
    print(os.path.join('runs', 'detect', EXPERIMENT_NAME))
    print("Your best model weights are typically in: ")
    print(os.path.join('runs', 'detect', EXPERIMENT_NAME, 'weights', 'best.pt'))

if __name__ == '__main__':
    train_yolo_model()