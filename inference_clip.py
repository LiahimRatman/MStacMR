import numpy as np
import torch
import torchvision
from PIL import Image


def inference_clip_one_image(image_path,
                             crop_labels,
                             model,
                             preprocess,
                             device,
                             from_txt=False,
                             labels_path=''):

    if from_txt:
        with open(labels_path + image_path.split('/')[-1].split('.')[0] + '.txt', 'r') as f:
            text = f.readlines()
        crops = [{
            'class_id': int(crop_label[0]),
            'x': float(crop_label[1]),
            'y': float(crop_label[2]),
            'w': float(crop_label[3]),
            'h': float(crop_label[4]),
        } for crop_label in [item.strip().split() for item in text]]
    else:
        crops = [{
            'class_id': int(crop_label[0]),
            'x': float(crop_label[1]),
            'y': float(crop_label[2]),
            'w': float(crop_label[3]),
            'h': float(crop_label[4]),
        } for crop_label in crop_labels]

    if not crops:
        return []

    imm = Image.open(image_path)
    images = []
    for crop in crops:
        im = torchvision.transforms.functional.crop(imm,
                                                    round(imm.size[1] * (crop['y'] - crop['h'] / 2)),
                                                    round(imm.size[0] * (crop['x'] - crop['w'] / 2)),
                                                    round(imm.size[1] * crop['h']),
                                                    round(imm.size[0] * crop['w']))
        images.append(im)

    images = [preprocess(image) for image in images]
    image_input = torch.tensor(np.stack(images)).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)

    return image_features


# todo add batch inference
