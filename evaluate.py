import pickle
import torch

from VSRN import VSRN
from dao import get_config
from data import get_dataloader
from search_service import encode_data, i2t, t2i, get_i2t_retrieval_scores, get_t2i_retrieval_scores
from Vocabulary import Vocabulary


def evaluate(model_path):
    # load model and options
    checkpoint = torch.load(model_path,
                            map_location="cpu")

    vocab = pickle.load(open('checkpoints_and_vocabs/f30k_precomp_vocab.pkl', 'rb'))
    # load vocabulary used by the model
    params = get_config('inference_config.yaml')
    params['vocab_size'] = len(vocab)
    model = VSRN(params['grad_clip'],
                 params['image_embedding_dim'],
                 params['gcn_embedding_size'],
                 params['vocab_size'],
                 params['caption_encoder_word_dim'],
                 params['caption_encoder_num_layers'],
                 params['caption_encoder_embedding_size'],
                 params['dim_vid'],  # todo вероятно это то же самое, что и gcn_embedding_size, но надо проверить
                 params['dim_caption_generation_hidden'],
                 params['input_dropout_p_caption_generation_enc'],
                 params['rnn_type_caption_generation_enc'],
                 params['rnn_dropout_p_caption_generation_enc'],
                 params['bidirectional_enc'],
                 params['max_caption_len'],
                 params['dim_word_caption_generation'],
                 params['input_dropout_p_caption_generation_dec'],
                 params['rnn_type_caption_generation_dec'],
                 params['rnn_dropout_p_caption_generation_dec'],
                 params['bidirectional_dec'],
                 params['margin'],
                 params['measure'],
                 params['max_violation'],
                 params['learning_rate'])

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    val_loader = get_dataloader(
        type='eval',
        annotations_map_name='checkpoints_and_vocabs/full_dataset_CTC_test_mapa_good.json',
        image_embeddings_name='precomputed_embeddings/final_all_test_emb_CLIP_fixed.npy',
        ocr_embeddings_name='precomputed_embeddings/final_all_test_emb_CLIP_fixed.npy',
        images_path='',
        vocab=vocab
    )

    print('Computing results...')
    img_embs, cap_embs = encode_data(model, val_loader)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    r, rt = get_i2t_retrieval_scores(img_embs,
                cap_embs,
                similarity_measure=params['measure'],
                return_ranks=True)
    ri, rti = get_t2i_retrieval_scores(img_embs,
                  cap_embs,
                  similarity_measure=params['measure'],
                  return_ranks=True)

    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.2f" % rsum)
    print("Average i2t Recall: %.2f" % ar)
    print("Image to text: %.2f %.2f %.2f %.2f %.2f" % r)
    print("Average t2i Recall: %.2f" % ari)
    print("Text to image: %.2f %.2f %.2f %.2f %.2f" % ri)

    # torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')


evaluate('checkpoints_and_vocabs/model_best1.pth.tar')
