import nltk
import numpy as np
import torch


def prepare_captions(caption, vocab, max_len):  # todo make here a function for caption encoding
    # Convert caption (string) to word ids.
    tokens = nltk.tokenize.word_tokenize(str(caption).lower())

    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))
    target = torch.Tensor(caption)

    # Deal with caption model data
    # label = np.zeros(self.max_len)
    mask = np.zeros(max_len + 1)
    gts = np.zeros((max_len + 1))

    # print(tokens)
    cap_caption = ['<start>'] + tokens + ['<end>']
    # print(cap_caption)
    if len(cap_caption) > max_len - 1:
        cap_caption = cap_caption[:max_len]
        cap_caption[-1] = '<end>'

    for j, w in enumerate(cap_caption):
        gts[j] = vocab(w)

    non_zero = (gts == 0).nonzero()

    mask[:int(non_zero[0][0]) + 1] = 1

    caption_label = torch.from_numpy(gts).type(torch.LongTensor)
    caption_mask = torch.from_numpy(mask).type(torch.FloatTensor)

    return target, caption_label, caption_mask


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    image_embeddings, captions, ids, img_ids, caption_labels, caption_masks, ocr_embeddings = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    image_embeddings = torch.stack(image_embeddings, 0)

    caption_labels_ = torch.stack(caption_labels, 0)
    caption_masks_ = torch.stack(caption_masks, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    ocr_embeddings = torch.stack(ocr_embeddings, 0)

    return image_embeddings, targets, lengths, ids, caption_labels_, caption_masks_, ocr_embeddings
