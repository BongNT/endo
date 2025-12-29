from ultralytics import YOLO

# Load a model
model = YOLO("yolo11l-seg.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/home/bongmedai/Endo/datasets/endo_coco_seg.yaml", epochs=1000, imgsz=640, cache=False)
