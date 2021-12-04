import pickle
# import os

# import torch

# import data
# from vocab import Vocabulary  # NOQA
from VSRN import VSRN
# from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data
# import tensorboard_logger as tb_logger
# import argparse

from metrics_utils import accuracy, validate
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


def main():  # todo refactor parameters
    # Hyper Parameters
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', default='data_big',
    #                     help='path to datasets')
    # parser.add_argument('--data_name', default='precomp',
    #                     help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    # parser.add_argument('--vocab_path', default='./vocab/',
    #                     help='Path to saved vocabulary pickle files.')
    # parser.add_argument('--margin', default=0.2, type=float,
    #                     help='Rank loss margin.')
    # parser.add_argument('--num_epochs', default=30, type=int,
    #                     help='Number of training epochs.')
    # parser.add_argument('--batch_size', default=128, type=int,
    #                     help='Size of a training mini-batch.')
    # parser.add_argument('--word_dim', default=300, type=int,
    #                     help='Dimensionality of the word embedding.')
    # parser.add_argument('--embed_size', default=2048, type=int,
    #                     help='Dimensionality of the joint embedding.')
    # parser.add_argument('--grad_clip', default=2., type=float,
    #                     help='Gradient clipping threshold.')
    # parser.add_argument('--crop_size', default=224, type=int,
    #                     help='Size of an image crop as the CNN input.')
    # parser.add_argument('--num_layers', default=1, type=int,
    #                     help='Number of GRU layers.')
    # parser.add_argument('--learning_rate', default=.0002, type=float,
    #                     help='Initial learning rate.')
    # parser.add_argument('--lr_update', default=15, type=int,
    #                     help='Number of epochs to update the learning rate.')
    # parser.add_argument('--workers', default=10, type=int,
    #                     help='Number of data loader workers.')
    # parser.add_argument('--log_step', default=10, type=int,
    #                     help='Number of steps to print and record the log.')
    # parser.add_argument('--val_step', default=500, type=int,
    #                     help='Number of steps to run validation.')
    # parser.add_argument('--logger_name', default='runs/runX',
    #                     help='Path to save the model and Tensorboard log.')
    # parser.add_argument('--resume', default='', type=str, metavar='PATH',
    #                     help='path to latest checkpoint (default: none)')
    # parser.add_argument('--max_violation', action='store_true',
    #                     help='Use max instead of sum in the rank loss.')
    # parser.add_argument('--img_dim', default=2048, type=int,
    #                     help='Dimensionality of the image embedding.')
    # parser.add_argument('--finetune', action='store_true',
    #                     help='Fine-tune the image encoder.')
    # parser.add_argument('--cnn_type', default='vgg19',
    #                     help="""The CNN used for image encoder
    #                     (e.g. vgg19, resnet152)""")
    # parser.add_argument('--use_restval', action='store_true',
    #                     help='Use the restval data for training on MSCOCO.')
    # parser.add_argument('--measure', default='cosine',
    #                     help='Similarity measure used (cosine|order)')
    # parser.add_argument('--use_abs', action='store_true',
    #                     help='Take the absolute value of embedding vectors.')
    # parser.add_argument('--no_imgnorm', action='store_true',
    #                     help='Do not normalize the image embeddings.')
    # parser.add_argument('--reset_train', action='store_true',
    #                     help='Ensure the training is always done in '
    #                     'train mode (Not recommended).')
    # ### AM Parameters
    # parser.add_argument('--text_number', default=15, type=int,
    #                     help='Number of ocr tokens used (max. 20).')
    # parser.add_argument('--text_dim', default=300, type=int,
    #                     help='Dimension of scene text embedding - default 300')
    #
    # ###caption parameters
    # parser.add_argument(
    #     '--dim_vid',
    #     type=int,
    #     default=2048,
    #     help='dim of features of video frames')
    # parser.add_argument(
    #     '--dim_hidden',
    #     type=int,
    #     default=512,
    #     help='size of the rnn hidden layer')
    # parser.add_argument(
    #     "--bidirectional",
    #     type=int,
    #     default=0,
    #     help="0 for disable, 1 for enable. encoder/decoder bidirectional.")
    # parser.add_argument(
    #     '--input_dropout_p',
    #     type=float,
    #     default=0.2,
    #     help='strength of dropout in the Language Model RNN')
    # parser.add_argument(
    #     '--rnn_type', type=str, default='gru', help='lstm or gru')
    #
    # parser.add_argument(
    #     '--rnn_dropout_p',
    #     type=float,
    #     default=0.5,
    #     help='strength of dropout in the Language Model RNN')
    #
    # parser.add_argument(
    #     '--dim_word',
    #     type=int,
    #     default=300,  # 512
    #     help='the encoding size of each token in the vocabulary, and the video.'
    # )
    # parser.add_argument(
    #     "--max_len",
    #     type=int,
    #     default=60,
    #     help='max length of captions(containing <sos>,<eos>)')
    #
    # opt = parser.parse_args()
    # print(opt)

    # Load Vocabulary Wrapper  # todo make new vocab
    vocab = pickle.load(open('checkpoints_and_vocabs/f30k_precomp_vocab.pkl', 'rb'))
    vocab_size = len(vocab)

    # Get data loaders
    # train_loader, val_loader = data.get_loaders(opt.data_path, opt.data_name, vocab, opt.crop_size, opt.batch_size,
    #                                             opt.workers, opt.use_restval, opt.max_len, opt.text_number, opt.text_dim)
    train_loader = get_dataloader(
        type='train',
        annotations_map_name='checkpoints_and_vocabs/full_dataset_train_mapa_good.json',
        image_embeddings_name='precomputed_embeddings/final_all_train_emb_CLIP_fp16.npy',
        ocr_embeddings_name='precomputed_embeddings/final_all_train_emb_CLIP_fp16.npy',
        images_path='',
        vocab=vocab
    )  # todo придумать параметры

    opt = None
    val_loader = None
    # Construct the model
    model = VSRN(opt)

    # Train the Model
    best_rsum = 0

    for epoch in range(opt.num_epochs):
        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        best_rsum = train(opt, train_loader, model, epoch, val_loader, best_rsum)

        # evaluate on validation set
        rsum = validate(opt, val_loader, model)  # todo подумать как реализовать разделение на вал сет. Пока не трогаю

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, prefix=opt.logger_name + '/')


def train(opt, train_loader, model, epoch, val_loader, best_rsum):  # todo add time tracking
    # switch to train mode
    model.train_start()
    for i, train_data in enumerate(train_loader):
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
        # if model.Eiters % opt.val_step == 0:
        #     # validate(opt, val_loader, model)
        #
        #     # evaluate on validation set
        #     rsum = validate(opt, val_loader, model)
        #
        #     # remember best R@ sum and save checkpoint
        #     is_best = rsum > best_rsum
        #     best_rsum = max(rsum, best_rsum)
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'model': model.state_dict(),
        #         'best_rsum': best_rsum,
        #         'opt': opt,
        #         'Eiters': model.Eiters,
        #     }, is_best, prefix=opt.logger_name + '/')

    return best_rsum


if __name__ == '__main__':
    main()