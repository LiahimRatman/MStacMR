import torch
import torch.nn as nn
import torch.nn.init
# import torchvision.models as models
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from GCN_lib.Rs_GCN import Rs_GCN

from search_service import l2norm


# from models import DecoderRNN, EncoderRNN, S2VTAttModel, S2VTModel


# todo Вот здесь нужно применить новые картинки
def EncoderImage(use_precomputed,
                 data_name,
                 img_dim,
                 embed_size,
                 finetune=False,
                 cnn_type='vgg19',
                 text_number=16,
                 text_dim=300,
                 use_abs=False,
                 no_imgnorm=False,
                 use_txt_emb=True):
    """A wrapper to image encoders. Chooses between an encoder that uses
    precomputed image features, `EncoderImagePrecomp`, or an encoder that
    computes image features on the fly `EncoderImageFull`.
    """
    if use_precomputed:
        if use_txt_emb == True:
            # USED FOR SCENE TEXT FEATURES
            img_enc = EncoderImagePrecompAttn(
                img_dim, embed_size, data_name, text_number, text_dim, use_abs, no_imgnorm)

    assert "Bad Decision"
    #     else:
    #         img_enc = EncoderImagePrecomp(
    #             img_dim, embed_size, use_abs, no_imgnorm)  # todo выпилить
    # else:
    #     img_enc = EncoderImageFull(
    #         embed_size, finetune, cnn_type, use_abs, no_imgnorm)  # todo переделать

    return img_enc


# tutorials/09 - Image Captioning
class EncoderImageFull(nn.Module):
    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
                 use_abs=False, no_imgnorm=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        # Load a pre-trained model
        self.cnn = self.get_cnn(cnn_type, True)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                                embed_size)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
            self.cnn.module.fc = nn.Sequential()

        self.init_weights()

    # def get_cnn(self, arch, pretrained):
    #     """Load a pretrained CNN and parallelize over GPUs
    #     """
    #     if pretrained:
    #         print("=> using pre-trained model '{}'".format(arch))
    #         model = models.__dict__[arch](pretrained=True)
    #     else:
    #         print("=> creating model '{}'".format(arch))
    #         model = models.__dict__[arch]()
    #
    #     if arch.startswith('alexnet') or arch.startswith('vgg'):
    #         model.features = nn.DataParallel(model.features)
    #         model.cuda()
    #     else:
    #         model = nn.DataParallel(model).cuda()
    #
    #     return model

    def load_state_dict(self, state_dict):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if 'cnn.classifier.1.weight' in state_dict:
            state_dict['cnn.classifier.0.weight'] = state_dict[
                'cnn.classifier.1.weight']
            del state_dict['cnn.classifier.1.weight']
            state_dict['cnn.classifier.0.bias'] = state_dict[
                'cnn.classifier.1.bias']
            del state_dict['cnn.classifier.1.bias']
            state_dict['cnn.classifier.3.weight'] = state_dict[
                'cnn.classifier.4.weight']
            del state_dict['cnn.classifier.4.weight']
            state_dict['cnn.classifier.3.bias'] = state_dict[
                'cnn.classifier.4.bias']
            del state_dict['cnn.classifier.4.bias']

        super(EncoderImageFull, self).load_state_dict(state_dict)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)

        # normalization in the image embedding space
        features = l2norm(features)

        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features


class EncoderImagePrecomp(nn.Module):
    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        # print(images)
        # images = images.view(images.size(0), 73728)
        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


# Encoder for precomputed embeddings
class EncoderImagePrecompAttn(nn.Module):
    def __init__(self,
                 img_dim,
                 embed_size,
                 data_name,  # todo выпилить data_name
                 text_number,
                 text_dim,
                 use_abs=False,
                 no_imgnorm=False):
        super(EncoderImagePrecompAttn, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.data_name = 'f30k_precomp'  # todo check here     data_name

        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

        # GSR
        self.img_rnn = nn.GRU(embed_size, embed_size, 1, batch_first=True)

        # GCN reasoning
        self.Rs_GCN_1 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_2 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_3 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_4 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)

        if self.data_name == 'f30k_precomp' or True:  # todo check here
            self.bn = nn.BatchNorm1d(embed_size)

        # FOR SCENE TEXT FEATURES
        self.bn_scene_text = nn.BatchNorm1d(text_number)
        self.fc_scene_text = nn.Linear(text_dim, embed_size)

        # GCN reasoning
        self.Text_GCN_1 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Text_GCN_2 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Text_GCN_3 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Text_GCN_4 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, scene_text):
        """Extract image feature vectors."""

        # IMAGE FEATURES
        fc_img_emd = self.fc(images)
        if self.data_name != 'f30k_precomp' and False:  # todo check here
            fc_img_emd = l2norm(fc_img_emd)

        # fc_img_emd = torch.cat((fc_img_emd, fc_scene_text), dim=1)

        # GCN reasoning
        # -> B,D,N
        GCN_img_emd = fc_img_emd.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_1(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_2(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_3(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_4(GCN_img_emd)
        # -> B,N,D
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)
        GCN_img_emd = l2norm(GCN_img_emd)
        visual_features = torch.mean(GCN_img_emd, dim=1)
        # rnn_img, hidden_state = self.img_rnn(GCN_img_emd)
        # visual_features = hidden_state[0]

        # SCENE TEXT FEATURES --- AM --------
        # fc_scene_text = self.bn_scene_text(scene_text)
        fc_scene_text = scene_text
        fc_scene_text = F.leaky_relu(self.fc_scene_text(fc_scene_text))
        fc_scene_text = l2norm(fc_scene_text)

        # Scene Text Reasoning
        # -> B,D,N
        GCN_scene_text_emd = fc_scene_text.permute(0, 2, 1)
        GCN_scene_text_emd = self.Text_GCN_1(GCN_scene_text_emd)
        GCN_scene_text_emd = self.Text_GCN_2(GCN_scene_text_emd)
        GCN_scene_text_emd = self.Text_GCN_3(GCN_scene_text_emd)
        GCN_scene_text_emd = self.Text_GCN_4(GCN_scene_text_emd)
        # # -> B,N,D
        GCN_scene_text_emd = GCN_scene_text_emd.permute(0, 2, 1)
        GCN_scene_text_emd = l2norm(GCN_scene_text_emd)
        fc_scene_text = torch.mean(GCN_scene_text_emd, dim=1)

        # FINAL AGGREGATION
        # fc_scene_text = torch.mean(fc_scene_text, dim=1)

        # features = torch.mul(visual_features, fc_scene_text) + visual_features  # todo return
        features = visual_features

        # features = torch.mean(GCN_img_emd, dim=1)

        if self.data_name == 'f30k_precomp' or True:  # todo check here
            features = self.bn(features)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features, GCN_img_emd

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecompAttn, self).load_state_dict(new_state)
