# import clip
import numpy as np
import torch
import torchvision
from PIL import Image


def inference_clip_one_image(image_path,
                             crop_labels,
                             model,
                             preprocess,
                             device):
    # model.cuda().eval()
    model.to(device)
    model.eval()
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    crops = [{
        'class_id': int(crop_label[0]),
        'x': float(crop_label[1]),
        'y': float(crop_label[2]),
        'w': float(crop_label[3]),
        'h': float(crop_label[4]),
    } for crop_label in crop_labels]
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

# model, preprocess = clip.load("ViT-B/32")
# emb = inference_clip_one_image('C:/Users/Mikhail Korotkov/PycharmProjects/MStacMR/VG/images/test/101.jpg',
#                          [(torch.tensor(445.), 0.3518750071525574, 0.6633333563804626, 0.32875001430511475, 0.20999999344348907)],
#                          model,
#                          preprocess,
#                          torch.device("cpu"))
#
# print(emb.shape)
