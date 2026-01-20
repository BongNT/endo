from ultralytics import YOLO

model = YOLO("/home/bongmedai/Endo/ultralytics/runs/detect/yolo_l_ade_carci/weights/best.pt")

results = model.val(data="/home/bongmedai/Endo/datasets/medai_endo_det/medai_endo_det.yaml", plots=True, save=True, save_txt=True)
# results = model.predict(
#     source="/home/bongmedai/Endo/datasets/endo_coco_crop/images/val",
#     save=True,
#     save_txt=True,
#     conf=0.5
# )

# print(results)