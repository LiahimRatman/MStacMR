from detect_yolov5 import run


def detect_image_regions_on_train_datasets(weights='YOLOv5_train/YOLOv5_model/weights/best.pt',
                                           imgsz=416,
                                           conf_thres=0.1):
    for dataset_name in ['CTC', 'flickr30k', 'text_caps']:
        run(weights=weights,
            imgsz=[imgsz, imgsz],
            conf_thres=conf_thres,
            source='STACMR_train/' + dataset_name + '/images',
            # source='VG/images/test',
            save_txt=True,
            # project='YOLOv5_train/res',
            project='STACMR_train/' + dataset_name,
            name='detections',
            nosave=True)


# import argparse
# import torch

# from train_yolov5 import train
# from utils.callbacks import Callbacks

# train(hyp='data\hyps\hyp.scratch.yaml',
#       opt=argparse.Namespace(adam=False, artifact_alias='latest', batch_size=64, bbox_interval=-1, bucket='', cache=False, cfg='',
#                              data='YOLOv5_train/VG_dataset.yaml', device='', entity=None, epochs=50, evolve=None, exist_ok=False,
#                              freeze=10, hyp='data\\hyps\\hyp.scratch.yaml', image_weights=False, imgsz=416, label_smoothing=0.0,
#                              linear_lr=False, local_rank=-1, multi_scale=False, name='YOLOv5_model', noautoanchor=False, nosave=False,
#                              noval=False, patience=100, project='YOLOv5_train', quad=False, rect=False, resume=False,
#                              save_dir='YOLOv5_train\\YOLOv5_model', save_period=-1, single_cls=False, sync_bn=False, upload_dataset=False,
#                              weights='yolov5l.pt', workers=-1),
#       device=torch.device("cpu"),
#       callbacks=Callbacks()
# )  # НЕ Работает из-за мультипроцесса


# run(weights='YOLOv5_train/YOLOv5_model/weights/best.pt',
# for dataset_name in ['flickr30k', 'text_caps', 'CTC']:
#     run(weights='yolo_best.pt',
#         imgsz=[416, 416],
#         conf_thres=0.1,
#         source='STACMR_train/CTC/images',
#         # source='VG/images/test',
#         save_txt=True,
#         # project='YOLOv5_train/res',
#         project='STACMR_train/CTC/labels',
#         name='detections',
#         nosave=True)


# print(argparse.Namespace(adam=False, artifact_alias='latest', batch_size=64, bbox_interval=-1, bucket='', cache='ram', cfg='', data='YOLOv5_train/VG_dataset.yaml', device='', entity=None, epochs=50, evolve=None, exist_ok=False, freeze=10, hyp='data\\hyps\\hyp.scratch.yaml', image_weights=False, imgsz=416, label_smoothing=0.0, linear_lr=False, local_rank=-1, multi_scale=False, name='YOLOv5_model', noautoanchor=False, nosave=False,
#                   noval=False, patience=100, project='YOLOv5_train', quad=False, rect=False, resume=False, save_dir='YOLOv5_train\\YOLOv5_model', save_period=-1, single_cls=False, sync_bn=False, upload_dataset=False,
#                   weights='yolov5l.pt', workers=8))

# train
# !python train_yolov5.py --img 416 --batch 64 --epochs 50 --data YOLOv5_train/VG_dataset.yaml --weights yolov5l.pt --cache --freeze 10 --project YOLOv5_train --name YOLOv5_model

# detect
# !python yolov5/detect_yolov5.py --weights YOLOv5_train/YOLOv5_model/weights/best.pt --img 416 --conf 0.1 --source STACMR_train/CTC/images --save-txt --project STACMR_train/labels/CTC --name detections --nosave
# !python yolov5/detect_yolov5.py --weights YOLOv5_train/YOLOv5_model/weights/best.pt --img 416 --conf 0.1 --source STACMR_train/flickr30k/images --save-txt --project STACMR_train/labels/flickr30k --nosave
# !python yolov5/detect_yolov5.py --weights YOLOv5_train/YOLOv5_model/weights/best.pt --img 416 --conf 0.1 --source STACMR_train/text_caps/images --save-txt --project STACMR_train/labels/text_caps --nosave

