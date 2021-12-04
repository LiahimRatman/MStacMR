# train
# !python yolov5/train.py --img 416 --batch 64 --epochs 50 --data YOLOv5_train/VG_dataset.yaml --weights yolov5l.pt --cache --freeze 10 --project YOLOv5_train --name YOLOv5_model

# detect
# !python yolov5/detect.py --weights YOLOv5_train/YOLOv5_model/weights/best.pt --img 416 --conf 0.1 --source STACMR_train/CTC/images --save-txt --project STACMR_train/labels/CTC --name detections --nosave
# !python yolov5/detect.py --weights YOLOv5_train/YOLOv5_model/weights/best.pt --img 416 --conf 0.1 --source STACMR_train/flickr30k/images --save-txt --project STACMR_train/labels/flickr30k --nosave
# !python yolov5/detect.py --weights YOLOv5_train/YOLOv5_model/weights/best.pt --img 416 --conf 0.1 --source STACMR_train/text_caps/images --save-txt --project STACMR_train/labels/text_caps --nosave

