import torch

from caption_models import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, xyxy2xywh, scale_coords

DEVICE = torch.device("cpu")
IMAGE_SIZE = 416
CONFIDENCE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.45


def inference_yolo_on_one_image(image_path,
                                model_path):

    device = torch.device("cpu")
    model = DetectMultiBackend(model_path, device=device, dnn=False)

    stride = model.stride
    imgsz = check_img_size(IMAGE_SIZE, s=stride)
    dataset = LoadImages(image_path, img_size=imgsz, stride=stride, auto=model.pt)
    model.warmup(imgsz=(1, 3, imgsz, imgsz),
                 half=False)  # warmup
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im, augment=False)

        # NMS
        pred = non_max_suppression(pred, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, None, False, max_det=36)
        det = pred[0]

        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class

        gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]
        # Write results
        results = []
        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh)
            results.append(line)

    return results


# todo add batch inference


# pred = inference_yolo_on_one_image('C:/Users/Mikhail Korotkov/PycharmProjects/MStacMR/VG/images/test/101.jpg',
#                                    'yolo_best.pt')
#
# print(pred)
