import pickle
from tqdm import tqdm

from VSRN import VSRN
from metrics_utils import validate
from dao import save_checkpoint
from data import get_dataloader
from train_utils import adjust_learning_rate


class Vocabulary(object):  # todo выпилить это
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def main():
    # Load Vocabulary Wrapper  # todo make new vocab
    vocab = pickle.load(open('checkpoints_and_vocabs/f30k_precomp_vocab.pkl', 'rb'))

    train_loader = get_dataloader(
        type='train',
        annotations_map_name='checkpoints_and_vocabs/full_dataset_train_mapa_good.json',
        image_embeddings_name='precomputed_embeddings/final_all_train_emb_CLIP_fp16.npy',
        ocr_embeddings_name='precomputed_embeddings/final_all_train_emb_CLIP_fp16.npy',
        images_path='',
        vocab=vocab
    )
    val_loader = get_dataloader(
        type='eval',
        annotations_map_name='checkpoints_and_vocabs/full_dataset_CTC_test_mapa_good.json',
        image_embeddings_name='precomputed_embeddings/final_all_test_emb_CLIP_fp16.npy',
        ocr_embeddings_name='precomputed_embeddings/final_all_test_emb_CLIP_fp16.npy',
        images_path='',
        vocab=vocab
    )

    num_epochs = 30
    # batch_size = 128
    grad_clip = 2.0
    gcn_embedding_size = 512
    image_embedding_dim = 512
    # data_name = 'precomp'
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

    # Train the Model
    best_rsum = 0

    for epoch in range(num_epochs):
        adjust_learning_rate(learning_rate,
                             lr_update,
                             model.optimizer,
                             epoch)

        # train for one epoch
        best_rsum = train(train_loader, model, epoch, val_loader, log_step, measure, best_rsum)
        print(best_rsum)
        # raise ValueError
        # evaluate on validation set
        rsum = validate(log_step,
                        measure,
                        val_loader,
                        model)  # todo подумать как реализовать разделение на вал сет. Пока не трогаю
        print(best_rsum)
        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum  # todo Что-то тут странно, чувствую, есть косяк
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            # 'opt': opt,
            # 'Eiters': model.Eiters,
        },
            is_best,
            prefix='runs/log/')


def train(train_loader, model, epoch, val_loader, log_step, measure, best_rsum):  # todo add time tracking
    # switch to train mode
    model.train_start()
    for i, train_data in tqdm(enumerate(train_loader)):
        model.train_start()

        # measure data loading time
        # data_time.update(time.time() - end)

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()
        #
        # Print log info
        # if model.Eiters % opt.log_step == 0:
        #     logging.info(
        #         'Epoch: [{0}][{1}/{2}]\t'
        #         '{e_log}\t'
        #         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #         .format(
        #             epoch, i, len(train_loader), batch_time=batch_time,
        #             data_time=data_time, e_log=str(model.logger)))
        #     print(
        #         'Epoch: [{0}][{1}/{2}]\t'
        #         '{e_log}\t'
        #         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #         .format(
        #             epoch, i, len(train_loader), batch_time=batch_time,
        #             data_time=data_time, e_log=str(model.logger)))

        # # validate at every val_step
        if i % 100 == 0 and i > 0:
            # validate(opt, val_loader, model)

            # evaluate on validation set
            rsum = validate(log_step,
                            measure,
                            val_loader,
                            model)

            # remember best R@ sum and save checkpoint
            is_best = rsum > best_rsum
            best_rsum = max(rsum, best_rsum)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                # 'opt': opt,
                # 'Eiters': model.Eiters,
            },
                is_best,
                prefix='runs/log/')

    return best_rsum


if __name__ == '__main__':
    main()
