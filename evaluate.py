import pickle
import torch

from VSRN import VSRN
from data import get_dataloader
from search_service import encode_data, i2t, t2i


def evaluate(model_path):
    # load model and options
    checkpoint = torch.load(model_path,
                            map_location="cpu")

    vocab = pickle.load(open('checkpoints_and_vocabs/f30k_precomp_vocab.pkl', 'rb'))
    # load vocabulary used by the model

    # construct model
    # batch_size = 128
    grad_clip = 2.0
    gcn_embedding_size = 512
    image_embedding_dim = 512
    caption_encoder_num_layers = 1
    vocab_size = len(vocab)
    caption_encoder_word_dim = 300  # caption embedding size
    caption_encoder_embedding_size = 512
    dim_vid = 512  # было 2048, подозреваю, что это много
    dim_caption_generation_hidden = 512  # мб теперь надо поменять
    input_dropout_p_caption_generation_enc = 0.2
    input_dropout_p_caption_generation_dec = 0.2
    rnn_type_caption_generation_enc = 'gru'
    rnn_type_caption_generation_dec = 'gru'
    rnn_dropout_p_caption_generation_enc = 0.5
    rnn_dropout_p_caption_generation_dec = 0.5
    bidirectional_enc = False
    bidirectional_dec = False
    max_caption_len = 60
    dim_word_caption_generation = 300  # output of encoder decoder embedding size
    margin = 0.2
    measure = 'cosine'
    max_violation = False
    learning_rate = 0.0002
    lr_update = 15
    log_step = 10

    # model = VSRN(opt)
    model = VSRN(grad_clip,
                 image_embedding_dim,
                 gcn_embedding_size,
                 vocab_size,
                 caption_encoder_word_dim,
                 caption_encoder_num_layers,
                 caption_encoder_embedding_size,
                 dim_vid,  # todo вероятно это то же самое, что и gcn_embedding_size, но надо проверить
                 dim_caption_generation_hidden,
                 input_dropout_p_caption_generation_enc,
                 rnn_type_caption_generation_enc,
                 rnn_dropout_p_caption_generation_enc,
                 bidirectional_enc,
                 max_caption_len,
                 dim_word_caption_generation,
                 input_dropout_p_caption_generation_dec,
                 rnn_type_caption_generation_dec,
                 rnn_dropout_p_caption_generation_dec,
                 bidirectional_dec,
                 margin,
                 measure,
                 max_violation,
                 learning_rate)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    val_loader = get_dataloader(
        type='eval',
        annotations_map_name='checkpoints_and_vocabs/full_dataset_CTC_test_mapa_good.json',
        image_embeddings_name='precomputed_embeddings/final_all_test_emb_CLIP_fp16.npy',
        ocr_embeddings_name='precomputed_embeddings/final_all_test_emb_CLIP_fp16.npy',
        images_path='',
        vocab=vocab
    )

    print('Computing results...')
    img_embs, cap_embs = encode_data(model, val_loader)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    # no cross-validation, full evaluation
    r, rt = i2t(img_embs,
                cap_embs,
                measure=measure,
                return_ranks=True)
    ri, rti = t2i(img_embs,
                  cap_embs,
                  measure=measure,
                  return_ranks=True)
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.2f" % rsum)
    print("Average i2t Recall: %.2f" % ar)
    print("Image to text: %.2f %.2f %.2f %.2f %.2f" % r)
    print("Average t2i Recall: %.2f" % ari)
    print("Text to image: %.2f %.2f %.2f %.2f %.2f" % ri)

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')
