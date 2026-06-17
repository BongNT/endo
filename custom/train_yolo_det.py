from ultralytics import YOLO


def main():
    # Load a model
    # Using YOLO11 Large as requested (pretrained on COCO)
    model = YOLO("yolo11l.pt")

    # Train the model
    # We use the configuration file that explicitly defines 'adenoma' (0) and 'carcinoma' (1).
    # single_cls=False ensures the model predicts multiple classes.
    model.train(
        data="/home/bongmedai/Endo/datasets/medai_endo_det/medai_endo_det.yaml",
        epochs=100,
        imgsz=640,
        project="/home/bongmedai/Endo/ultralytics/runs/detect",
        name="yolo_det_multiclass",  # Renamed to reflect multiclass nature
        device="0",
        patience=50,
        batch=16,
        save=True,
        cache=True,
        single_cls=False,  # Explicitly ensuring multiclass
    )


if __name__ == "__main__":
    main()
