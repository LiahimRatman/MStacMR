import os
from pathlib import Path

import torch
import torch.nn as nn
# import torch.nn.init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import AutoModel, AutoTokenizer

from train_utils import l2norm

# class LABSETextEncoder(transformers.models.bert.modeling_bert.BertModel):
class TextEncoder(nn.Module):
    def __init__(
        self,
        encoder_path: str,
        freeze_encoder = True,
        output_dim = 512,
        max_caption_len = 100,
        *args, **kwargs,
    ):
        super(TextEncoder, self).__init__()
        self.encoder_path = Path(os.path.abspath(encoder_path.rstrip('/')))
        # self.encoder_path = encoder_path
        # print(self.encoder_path)
        self.encoder = AutoModel.from_pretrained(self.encoder_path)
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(self.encoder_path)
        self.max_caption_len = max_caption_len

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.fc = nn.Linear(
            self.encoder.config.to_dict()['pooler_fc_size'],
            output_dim,
        )
        self.activation = nn.ReLU()
    
    def forward(self, encoder_input):
        """
        `encoded_input` should be something that comes after calling something like
        tokenizer(sentences, padding=True, truncation=True, max_length=64, return_tensors='pt')
        """

        encoder_input = self.encoder_tokenizer(
            encoder_input, ### list of strings
            max_length = self.max_caption_len,
            padding = True,#'max_length',
            truncation = True,
            return_tensors='pt',
        ).to(self.encoder.device)

        # tokenizer_outputs.to(self.device)

        encoder_output = self.encoder(**encoder_input) ### lots of stuff
        embeddings = encoder_output.pooler_output ### batch_size x hid_dim
        embeddings = self.fc(embeddings) ### batch_size x hid_dim
        embeddings = self.activation(embeddings)
        return embeddings


# RNN Based Language Model
# Это энкодер для кепшнов, тут будет Labse
class EncoderText(nn.Module):
    def __init__(
            self,
            vocab_size,
            word_dim,
            embed_size,
            num_layers,
            use_abs=False,
            device='cpu',
    ):
        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)  # todo Вот здесь мы хотим Labse

        self.init_weights()

        self.device = device

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def to(self, device):
        self = super(EncoderText, self).to(device)
        self.device = device
        return self

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size) - 1).to(self.device)
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out
