from ultralytics import YOLO

model = YOLO("/home/bongmedai/Endo/ultralytics/runs/segment/train8/weights/best.pt")

results = model.val(data="/home/bongmedai/Endo/datasets/medai_endo_data.yaml", plots=True, save=True, save_txt=True)
# results = model.predict(
#     source="/home/bongmedai/Endo/datasets/endo_coco_crop/images/val",
#     save=True,
#     save_txt=True,
#     conf=0.5
# )

# print(results)
