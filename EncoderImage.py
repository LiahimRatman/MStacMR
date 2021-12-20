import torch
import torch.nn as nn
#import torch.nn.init
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from GCN_lib.Rs_GCN import Rs_GCN

from train_utils import l2norm


def EncoderImage(
    use_precomputed,
    img_dim,
    embed_dim,
    num_ocr_boxes = 16,
    ocr_box_dim = 300,
    use_abs = False,
    use_bn = False,
    use_l2norm = False,
    use_l2norm_final = False,
    use_ocr_emb = True,
use_gcn_scene_text_emb=False
):
    """
    Wrapper for image encoders
    Chooses between an encoder that uses precomputed image features - `EncoderImagePrecomp`,
    or an encoder that computes image features on the fly - `EncoderImageFull`.
    """
    if use_precomputed:
        if use_ocr_emb:
            # USED FOR SCENE TEXT FEATURES
            encoder = EncoderImagePrecompAttn(
                img_dim = img_dim,
                embed_dim = embed_dim,
                num_ocr_boxes = num_ocr_boxes,
                ocr_box_dim = ocr_box_dim,
                use_abs = use_abs,
                use_l2norm = use_l2norm,
                use_l2norm_final = use_l2norm_final,
                use_ocr_emb = use_ocr_emb,
                use_gcn_scene_text_emb = use_gcn_scene_text_emb
            )
        else:
           raise ValueError('should initialize only with ocr features usage') 
    else:
        raise NotImplementedError('only precomp features encoder available for now')
    
    return encoder


# Encoder for precomputed embeddings
class EncoderImagePrecompAttn(nn.Module):
    def __init__(
        self,
        img_dim,
        embed_dim,
        num_ocr_boxes,
        ocr_box_dim,
        use_abs = False,
        use_bn = False,
        use_l2norm = False,
        use_l2norm_final = False,
        use_ocr_emb = True,
        use_gcn_scene_text_emb= False
    ):
        super(EncoderImagePrecompAttn, self).__init__()
        self.embed_dim = embed_dim
        self.use_abs = use_abs       
        self.use_bn = use_bn
        self.use_l2norm = use_l2norm
        self.use_l2norm_final = use_l2norm_final
        self.use_ocr_emb = use_ocr_emb
        self.use_gcn_scene_text_emb = use_gcn_scene_text_emb

        self.fc = nn.Linear(img_dim, embed_dim)

        self.init_weights()

        # GSR
        self.img_rnn = nn.GRU(embed_dim, embed_dim, 1, batch_first=True)

        # GCN reasoning
        self.Rs_GCN_1 = Rs_GCN(in_channels=embed_dim, inter_channels=embed_dim)
        self.Rs_GCN_2 = Rs_GCN(in_channels=embed_dim, inter_channels=embed_dim)
        self.Rs_GCN_3 = Rs_GCN(in_channels=embed_dim, inter_channels=embed_dim)
        self.Rs_GCN_4 = Rs_GCN(in_channels=embed_dim, inter_channels=embed_dim)

        self.bn = nn.BatchNorm1d(embed_dim)

        # FOR SCENE TEXT FEATURES
        self.bn_scene_text = nn.BatchNorm1d(num_ocr_boxes)
        self.fc_scene_text = nn.Linear(ocr_box_dim, embed_dim)

        # GCN reasoning
        self.Text_GCN_1 = Rs_GCN(in_channels=embed_dim, inter_channels=embed_dim)
        self.Text_GCN_2 = Rs_GCN(in_channels=embed_dim, inter_channels=embed_dim)
        self.Text_GCN_3 = Rs_GCN(in_channels=embed_dim, inter_channels=embed_dim)
        self.Text_GCN_4 = Rs_GCN(in_channels=embed_dim, inter_channels=embed_dim)

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, scene_text):
        """Extract image feature vectors."""

        # IMAGE FEATURES
        fc_img_emd = self.fc(images)

        # if self.use_l2norm:
        #     fc_img_emd = l2norm(fc_img_emd)

        ### WTF is this here?
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
        if self.use_l2norm:
            GCN_img_emd = l2norm(GCN_img_emd)
        visual_features = torch.mean(GCN_img_emd, dim=1)
        ### using AvgPool above instead rnn
        # rnn_img, hidden_state = self.img_rnn(GCN_img_emd)
        # visual_features = hidden_state[0]

        # SCENE TEXT FEATURES 
        if self.use_bn:
            fc_scene_text = self.bn_scene_text(scene_text)
        else:
            fc_scene_text = scene_text
            
        fc_scene_text = F.leaky_relu(self.fc_scene_text(fc_scene_text))
        if self.use_l2norm:
            fc_scene_text = l2norm(fc_scene_text)

        if self.use_gcn_scene_text_emb:
            # Scene Text Reasoning
            # -> B,D,N
            GCN_scene_text_emd = fc_scene_text.permute(0, 2, 1)
            GCN_scene_text_emd = self.Text_GCN_1(GCN_scene_text_emd)
            GCN_scene_text_emd = self.Text_GCN_2(GCN_scene_text_emd)
            GCN_scene_text_emd = self.Text_GCN_3(GCN_scene_text_emd)
            GCN_scene_text_emd = self.Text_GCN_4(GCN_scene_text_emd)
            # # -> B,N,D
            GCN_scene_text_emd = GCN_scene_text_emd.permute(0, 2, 1)
            if self.use_l2norm:
                GCN_scene_text_emd = l2norm(GCN_scene_text_emd)
            fc_scene_text = torch.mean(GCN_scene_text_emd, dim=1)
        
        ### this is weird too
        fc_scene_text = torch.mean(fc_scene_text, dim=1)

        # FINAL AGGREGATION
        if self.use_ocr_emb:
            ### why multiplying..?
            features = torch.mul(visual_features, fc_scene_text) + visual_features
        else:
            features = visual_features

        if self.use_bn:
            features = self.bn(features)

        # normalize in the joint embedding space
        if self.use_l2norm_final:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        ### ocr-blended features & pure gcn image features
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
