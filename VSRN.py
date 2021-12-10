import numpy as np
import torch

class VSRN:
    """
    """
    def __init__(
        self,
        image_encoder,
        text_encoder,
        caption_model,
        retrieval_criterion,
        caption_criterion,
        weight_retrieval_loss,
        weight_caption_loss,
        learning_rate,
        grad_clip,   
        device = 'cpu',
    ):
        
        self.device = device
        
        ### makes fusion of bboxes + ocr-encoded
        self.image_encoder = image_encoder.to(self.device)
        ### encodes caption inputs
        self.text_encoder = text_encoder.to(self.device)
        ### generates caption from hidden states
        self.caption_model = caption_model.to(self.device)
 
        # losses and optimizer
        self.retrieval_criterion = retrieval_criterion.to(self.device)
        self.caption_criterion = caption_criterion 
        self.weight_retrieval_loss = weight_retrieval_loss
        self.weight_caption_loss = weight_caption_loss

        # self.params = params
        self.optimizer = torch.optim.Adam(self.get_params(), lr=learning_rate)
        ### scalar on gradient vector norm
        self.grad_clip = grad_clip

        ### they're already in caption_model - that was causing below warning
        ### "UserWarning: optimizer contains a parameter group with duplicate parameters;
        ### in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967 for more information"
        # params += list(self.decoder.parameters())
        # params += list(self.encoder.parameters())

    def get_params(self):
        params = []
        params += list(self.image_encoder.parameters())
        params += list(self.text_encoder.parameters())
        params += list(self.caption_model.parameters())
        return params

    def state_dict(self):
        state_dict = [
            self.image_encoder.state_dict(),
            self.text_encoder.state_dict(),
            self.caption_model.state_dict(),
        ]
        return state_dict

    def load_state_dict(self, state_dict):
        self.image_encoder.load_state_dict(state_dict[0])
        self.text_encoder.load_state_dict(state_dict[1])
        if len(state_dict) > 2:
            self.caption_model.load_state_dict(state_dict[2])

    def train(self):
        """switch to train mode"""
        self.image_encoder.train()
        self.text_encoder.train()
        self.caption_model.train()

    def eval(self):
        """switch to evaluate mode"""
        self.image_encoder.eval()
        self.text_encoder.eval()
        self.caption_model.eval()

    def to(self, device):
        self.image_encoder.to(device)
        self.text_encoder.to(device)
        self.caption_model.to(device)
        ### !!!
        self.retrieval_criterion.to(device)

        self.device = device

        return self

    def calculate_caption_loss(self, fc_feats, labels, masks):

        labels = labels.to(self.device)
        masks = masks.to(self.device)

        seq_probs, _ = self.caption_model(fc_feats, labels, 'train')
        loss = self.caption_criterion(seq_probs, labels[:, 1:], masks[:, 1:])

        return loss

    def forward_emb(self, images, captions, lengths, scene_texts):
        """
        returns
        fusion of image and ocr features
        caption embeddings
        pure image embeddings
        """
        
        images = images.to(self.device)
        captions = captions.to(self.device)
        scene_texts = scene_texts.to(self.device)

        cap_emb = self.text_encoder(captions, lengths)
        img_emb, GCN_img_emd = self.image_encoder(images, scene_texts)

        return img_emb, cap_emb, GCN_img_emd

    def make_train_step(self, images, captions, lengths, ids, caption_labels, caption_masks, scene_texts):
        """training step"""

        # compute the embeddings
        img_emb, cap_emb, GCN_img_emd = self.forward_emb(images, captions, lengths, scene_texts)

        caption_loss = self.calculate_caption_loss(GCN_img_emd, caption_labels, caption_masks)
        # todo Тут бы выяснить, почему одно, а не другое
        # caption_loss = self.calculate_caption_loss(img_emb, caption_labels, caption_masks)

        retrieval_loss = self.retrieval_criterion(img_emb, cap_emb)

        loss = (
            self.weight_retrieval_loss * retrieval_loss
            +
            self.weight_caption_loss * caption_loss
        )

        # print(f"Loss: {loss}, caption loss: {caption_loss}, retrieval loss: {retrieval_loss}")

        # compute gradient and make optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            params = self.get_params()
            # for p in params:
            #     print(p.device)
            torch.nn.utils.clip_grad.clip_grad_norm_(params, self.grad_clip)
        self.optimizer.step()
