import torch
import torch.nn as nn

# from yolov5.models.common import DetectMultiBackend
from yolov5.models.yolo import Detect, Model
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression
from yolov5.models.experimental import Ensemble, Conv

DEVICE = torch.device("cpu")


def attempt_load(weights, map_location=None, inplace=True, fuse=True):
    # from yolov5.models.yolo import Detect, Model

    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    print('!')
    ckpt = torch.load(weights, map_location=map_location)  # load
    print('!!')
    if fuse:
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
    else:
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval())  # without layer fuse

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = inplace  # pytorch 1.7.0 compatibility
            if type(m) is Detect:
                if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model  # return ensemble


def inference_yolo_on_one_image(image_path,
                                model_path):
    device = torch.device("cpu")
    # model = DetectMultiBackend(model_path, device=device, dnn=False)

    # DetectMultiBackend
    model = attempt_load(model_path, map_location=device)

    stride = model.stride
    imgsz = check_img_size(416, s=stride)
    dataset = LoadImages(image_path, img_size=imgsz, stride=stride, auto=model.pt)

    model.warmup(imgsz=(1, 3, *imgsz), half=False)  # warmup
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im, augment=False)

        # NMS
        pred = non_max_suppression(pred, 0.1, 0.45, None, False, max_det=36)

    return pred


# ckpt = torch.load('yolo_best.pt')

pred = inference_yolo_on_one_image('VG/images/test/101.jpg',
                                   'yolo_best.pt')

# print(pred)
# print(ckpt)