import numpy as np
import torch
from tqdm import tqdm


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """

    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()

    return score


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)

    return X


def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
    index_list = []

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                if torch.cuda.is_available():
                    d2 = order_sim(torch.Tensor(im2).cuda(),
                                   torch.Tensor(captions).cuda())
                else:
                    d2 = order_sim(torch.Tensor(im2),
                                   torch.Tensor(captions))
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            d = np.dot(im, captions.T).flatten()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
    ims = np.array([images[i] for i in range(0, len(images), 5)])

    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # Compute scores
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                if torch.cuda.is_available():
                    d2 = order_sim(torch.Tensor(ims).cuda(),
                                   torch.Tensor(q2).cuda())
                else:
                    d2 = order_sim(torch.Tensor(ims),
                                   torch.Tensor(q2))
                d2 = d2.cpu().numpy()

            d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = np.dot(queries, ims.T)
        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


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


def encode_data(model, data_loader, log_step=10):
    """
    Encode all images and captions loadable by `data_loader`
    Is used for validation
    """
    # batch_time = AverageMeter()
    # val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    # end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    # image (Tensor), caption (Encoded by NLTK), caption length, image id, caption (Encoded and with padding up to max_len), caption mask (the same but mask), scene text (Encoded, 15 texts per image with emb len 300)
    for (images, captions, lengths, ids, caption_labels, caption_masks, scene_text) in tqdm(data_loader):
        # make sure val logger is used
        # model.logger = val_logger

        # compute the embeddings
        img_emb, cap_emb, GCN_img_emd = model.forward_emb(images, captions, lengths, scene_text)

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

        img_embs[ids, :] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids, :] = cap_emb.data.cpu().numpy().copy()

        # del images, captions

    return img_embs, cap_embs
