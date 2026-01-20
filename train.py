from ultralytics import YOLO

# Load a model
model = YOLO("/home/bongmedai/Endo/yolo11l-seg.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data="/home/bongmedai/Endo/datasets/medai_endo_data.yaml",
    epochs=500,
    imgsz=640,
    # cfg="/home/bongmedai/Endo/ultralytics/custom/hyp_medical.yaml",
    cache=False, 
    single_cls=True
)
model.info()
