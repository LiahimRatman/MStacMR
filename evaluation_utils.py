# import pickle
import torch
import numpy as np
# from tqdm import tqdm
from tqdm.auto import tqdm


# def encode_data(model, dataloader, device='cpu'):
def encode_data(model, dataloader):
    """
    encodes all images and captions from `dataloader`
    used for validation
    """

    # switch to evaluate mode
    model.eval()
    # model = model.to(device)

    # numpy arrays to keep all the embeddings
    img_embs, cap_embs = None, None

    with torch.no_grad():
        ### batch of
        # image bboxes embeddings (N_bb)
        # caption tokens' indices full length,
        # caption length,
        # image id,
        # caption tokens' indices (truncated of padded up to max_len),
        # caption mask (the same but mask),
        # scene text embeddings (N_st)
        for (ids, images, ocr_features, tokenizer_outputs, captions, lengths, caption_labels, caption_masks) in tqdm(dataloader):

            # images = images.to(device)
            # captions = captions.to(device)
            # tokenizer_outputs = tokenizer_outputs.to(device)
            # scene_texts = scene_texts.to(device)

            # compute the embeddings
            img_emb, cap_emb, GCN_img_emb = model.forward_emb(images, ocr_features, tokenizer_outputs, captions, lengths)
            # initialize the numpy arrays given the size of the embeddings
            if img_embs is None:
                img_embs = np.zeros((len(dataloader.dataset), img_emb.size(1)))
            if cap_embs is None:
                cap_embs = np.zeros((len(dataloader.dataset), cap_emb.size(1)))

            img_embs[ids, :] = img_emb.data.cpu().numpy().copy()
            cap_embs[ids, :] = cap_emb.data.cpu().numpy().copy()

    return img_embs, cap_embs


def evaluate(model, dataloader, similarity_measure='cosine', captions_per_image=5):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs = encode_data(model, dataloader,)
    # np.save("img_emb.npy", img_embs[::5, ...])
    # np.save("cap_embs.npy", cap_embs)

    # caption retrieval
    eval_res_i2t, ranks_i2t = get_i2t_retrieval_scores(
        img_embs, cap_embs,
        similarity_measure=similarity_measure,
        captions_per_image=captions_per_image,
        return_ranks=True,
    )
    # image retrieval
    eval_res_t2i, ranks_t2i = get_t2i_retrieval_scores(
        img_embs, cap_embs,
        similarity_measure=similarity_measure,
        captions_per_image=captions_per_image,
        return_ranks=True,
        return_aggregated=False,
    )

    # (r1, r5, r10, medr, meanr) = eval_res_i2t
    # (r1i, r5i, r10i, medri, meanr) = eval_res_t2i

    average_recall_i2t = sum(eval_res_i2t[:3]) / 3
    average_recall_t2i = sum(eval_res_t2i[:3]) / 3
    average_recall = (average_recall_i2t + average_recall_t2i) / 2

    print('-' * 40)
    print("Image2Text: R@1: %.2f, R@5: %.2f, R@10: %.2f, median rank: %.2f, mean rank: %.2f" % eval_res_i2t)
    print("Average Image2Text Recall: %.2f" % average_recall_i2t)
    print("Text2Image: R@1: %.2f, R@5: %.2f, R@10: %.2f, median rank: %.2f, mean rank: %.2f" % eval_res_t2i)
    print("Average Text2Image Recall: %.2f" % average_recall_t2i)
    print("Average Recall: %.2f" % average_recall)
    print('-' * 40)

    # avg of recalls @1, @5 to be used for early stopping
    current_score = sum(eval_res_i2t[:2] + eval_res_t2i[:2]) / 4

    return current_score


def get_i2t_retrieval_scores(images, captions, similarity_measure='cosine', captions_per_image=5, return_ranks=False):
    """
    """
    n_unique_images = int(images.shape[0] / captions_per_image)

    ranks = np.zeros(n_unique_images)
    top1 = np.zeros(n_unique_images)

    for index in tqdm(range(n_unique_images)):
        # get query image (every 'captions_per_image'th item in embeddings)
        queries = images[captions_per_image * index].reshape(1, images.shape[1])

        # similarities
        if similarity_measure == 'cosine':
            d = np.dot(queries, captions.T).flatten()  ### (n_captions_total, )
        else:
            raise NotImplementedError

        # returns the indices that would sort an array decreasing
        indices_decreasing = np.argsort(d)[::-1]

        # ranking - for all true captions related to this image selecting minimal rank
        rank = np.inf
        for i in range(captions_per_image * index, captions_per_image * (index + 1), 1):
            ### looking how high the current caption index in the argsorted indices
            current_rank = np.where(indices_decreasing == i)[0][0]
            ### actual position of relevant caption in all ranked captions
            ### taking the highest among all captions_per_image relevant
            if current_rank < rank:
                rank = current_rank
        if np.isinf(rank):
            raise ValueError("how did that happen?")

        ranks[index] = rank  ### highest rank for relevant captions for image
        top1[index] = indices_decreasing[0]  ### which was the actual closest caption

    # compute recall@k metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    ### +1 cause indexation above from 0
    median_rank = np.floor(np.median(ranks)) + 1
    mean_rank = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, median_rank, mean_rank), (ranks, top1)
    return (r1, r5, r10, median_rank, mean_rank)


def get_t2i_retrieval_scores(images, captions, similarity_measure='cosine', captions_per_image=5, return_ranks=False,
                             return_aggregated=False):
    n_unique_images = int(images.shape[0] / captions_per_image)
    images_unique = np.array([images[i] for i in range(0, len(images), captions_per_image)])

    ranks_aggregated = np.zeros(n_unique_images)
    ranks_singles = np.zeros(captions_per_image * n_unique_images)
    top1 = np.zeros(captions_per_image * n_unique_images)

    for index in tqdm(range(n_unique_images)):

        # for every image get relevant query captions
        queries = captions[captions_per_image * index: captions_per_image * (index + 1)]

        # similarities between every of captions_per_image caption and all images
        if similarity_measure == 'cosine':
            d = np.dot(queries, images_unique.T)
        else:
            raise NotImplementedError

        indices = np.zeros(d.shape)

        for i in range(len(indices)):
            # returns the indices that would sort an array decreasing
            indices[i] = np.argsort(d[i])[::-1]
            ### looking how high the current image index in the argsorted indices
            ranks_singles[captions_per_image * index + i] = np.where(indices[i] == index)[0][0]
            top1[captions_per_image * index + i] = indices[i][0]

        ranks_aggregated[index] = np.min(ranks_singles[captions_per_image * index: captions_per_image * (index + 1)])

    # compute recall@k metrics
    if return_aggregated:
        ranks = ranks_aggregated
    else:
        ranks = ranks_singles

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    ### +1 cause indexation above from 0
    median_rank = np.floor(np.median(ranks)) + 1
    mean_rank = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, median_rank, mean_rank), (ranks, top1)
    return (r1, r5, r10, median_rank, mean_rank)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
