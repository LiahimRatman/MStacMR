import torch
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.autograd import Variable

import train_utils as utils

from ContrastiveLoss import ContrastiveLoss
from models import DecoderRNN, EncoderRNN, S2VTAttModel
from EncoderImage import EncoderImage
from EncoderText import EncoderText


# todo Вот тут много работы
class VSRN(object):
    """
    rkiros/uvs model
    """
    def __init__(self, opt):
        # tutorials/09 - Image Captioning
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(use_precomputed=True,
                                    data_name=opt.data_name,
                                    img_dim=opt.img_dim,
                                    embed_size=opt.embed_size,
                                    finetune=opt.finetune,
                                    cnn_type=opt.cnn_type,
                                    text_number=opt.text_number,
                                    text_dim=opt.text_dim,
                                    use_abs=opt.use_abs,
                                    no_imgnorm=opt.no_imgnorm)

        self.txt_enc = EncoderText(vocab_size=opt.vocab_size,
                                   word_dim=opt.word_dim,
                                   embed_size=opt.embed_size,
                                   num_layers=opt.num_layers,
                                   use_abs=opt.use_abs)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        #   captioning elements

        self.encoder = EncoderRNN(
            opt.dim_vid,
            opt.dim_hidden,
            bidirectional=False,#opt.bidirectional,
            input_dropout_p=opt.input_dropout_p,
            rnn_cell=opt.rnn_type,
            rnn_dropout_p=opt.rnn_dropout_p)
        self.decoder = DecoderRNN(
            opt.vocab_size,
            opt.max_len,
            opt.dim_hidden,
            opt.dim_word,
            input_dropout_p=opt.input_dropout_p,
            rnn_cell=opt.rnn_type,
            rnn_dropout_p=opt.rnn_dropout_p,
            bidirectional=opt.bidirectional)

        self.caption_model = S2VTAttModel(self.encoder, self.decoder)  # todo Вот тут мы хотим что-то другое для image captioning

        self.crit = utils.LanguageModelCriterion()  # todo Надо разобраться
        self.rl_crit = utils.RewardCriterion()  # todo Похоже, что не используется

        if torch.cuda.is_available():
            self.caption_model.cuda()

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         measure=opt.measure,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.decoder.parameters())
        params += list(self.encoder.parameters())
        params += list(self.caption_model.parameters())

        if opt.finetune:
            params += list(self.img_enc.cnn.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def calcualte_caption_loss(self, fc_feats, labels, masks):

        # labels = Variable(labels, volatile=False)
        # masks = Variable(masks, volatile=False)

        # torch.cuda.synchronize()
        labels = labels#.cuda()
        masks = masks#.cuda()

        # if torch.cuda.is_available():
        #     labels.cuda()
        #     masks.cuda()

        seq_probs, _ = self.caption_model(fc_feats, labels, 'train')
        loss = self.crit(seq_probs, labels[:, 1:], masks[:, 1:])

        return loss

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, scene_text, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset

        images = Variable(images)
        captions = Variable(captions)
        scene_text = Variable(scene_text)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            scene_text = scene_text.cuda()

        # Forward

        cap_emb = self.txt_enc(captions, lengths)  # Эмбеддинги кэпшнов
        img_emb, GCN_img_emd = self.img_enc(images, scene_text)  # img_emb - Зафьюженные результаты OCR и object detection

        return img_emb, cap_emb, GCN_img_emd

    def forward_loss(self, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb)
        return loss

    def train_emb(self, images, captions, lengths, ids, caption_labels, caption_masks, scene_text, *args):
        """One training step given images and captions.
        """

        # compute the embeddings
        img_emb, cap_emb, GCN_img_emd = self.forward_emb(images, captions, lengths, scene_text)

        # calculate captioning loss
        self.optimizer.zero_grad()
        caption_loss = self.calcualte_caption_loss(GCN_img_emd, caption_labels, caption_masks)  # todo Тут бы выяснить, почему одно, а не другое
        # caption_loss = self.calcualte_caption_loss(img_emb, caption_labels, caption_masks)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        retrieval_loss = self.forward_loss(img_emb, cap_emb)

        loss = 2.0 * retrieval_loss + caption_loss
        print(f"Loss: {loss}, caption loss: {caption_loss}, retrieval loss: {retrieval_loss}")

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
