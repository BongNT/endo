from ultralytics import YOLO

model = YOLO("/home/bongmedai/Endo/ultralytics/runs/segment/train5/weights/best.pt")

results = model.val(data="/home/bongmedai/Endo/datasets/endo_coco_seg.yaml", plots=True)
# results = model.predict(
#     source="/home/bongmedai/Endo/datasets/endo_coco_seg/images/val",
#     save=True,
#     save_txt=True,
#     conf=0.5
# )

# print(results)