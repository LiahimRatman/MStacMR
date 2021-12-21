from yolov5.detect import run


def detect_image_regions_on_train_datasets(weights='../YOLOv5_train/YOLOv5_model/weights/best.pt',
                                           imgsz=416,
                                           conf_thres=0.1):
    for dataset_name in ['CTC', 'flickr30k', 'text_caps']:
        run(weights=weights,
            imgsz=[imgsz, imgsz],
            conf_thres=conf_thres,
            source='../STACMR_train/' + dataset_name + '/images',
            # source='VG/images/test',
            save_txt=True,
            # project='YOLOv5_train/res',
            project='../STACMR_train/' + dataset_name,
            name='detections',
            nosave=True)
