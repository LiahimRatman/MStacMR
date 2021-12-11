import os
import torch

import pickle
from Vocabulary import Vocabulary
from EncoderImage import EncoderImage
from EncoderText import EncoderText, TextEncoder
from caption_models.DecoderRNN import DecoderRNN
from caption_models.EncoderRNN import EncoderRNN
from caption_models.S2VTAttModel import S2VTAttModel
from VSRN import VSRN
from train_utils import LanguageModelCriterion, ContrastiveLoss#, RewardCriterion

# from utilities import get_config


def create_model_from_config(config):
    ### device
    device = config.get('device')
    if not device or device == 'cuda':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ### load vocabulary used by the model
    vocab = pickle.load(open(config['training_params']['vocab_path'], 'rb'))

    ### create image encoder
    image_encoder = EncoderImage(
        use_precomputed=True,
        img_dim=config['image_encoder_params']['image_embedding_dim'],
        embed_dim=config['image_encoder_params']['gcn_embedding_dim'],
        num_ocr_boxes=config['image_encoder_params']['num_ocr_boxes'],
        ocr_box_dim=config['image_encoder_params']['ocr_box_dim'],
        use_abs=config['image_encoder_params']['use_abs'],
        use_bn=config['image_encoder_params']['use_bn'],
        use_l2norm=config['image_encoder_params']['use_l2norm'],
        use_l2norm_final=config['image_encoder_params']['use_l2norm_final'],
        use_ocr_emb=config['image_encoder_params']['use_ocr_emb'],
    )

    # text_encoder = EncoderText(
    #     vocab_size=len(vocab),
    #     word_dim=config['caption_encoder_params']['caption_encoder_word_dim'],
    #     embed_size=config['caption_encoder_params']['caption_encoder_embedding_dim'],
    #     num_layers=config['caption_encoder_params']['caption_encoder_num_layers'],
    #     device=device,
    # )

    text_encoder = TextEncoder(
        encoder_path=config['caption_encoder_params']['encoder_path'],
        output_dim=config['caption_encoder_params']['output_dim'],
        max_caption_len=config['caption_encoder_params']['max_caption_len'],
    )

    caption_model = S2VTAttModel(
        EncoderRNN(
            dim_vid=config['caption_generator_params']['dim_vid'],
            dim_hidden=config['caption_generator_params']['dim_caption_generation_hidden'],
            input_dropout_p=config['caption_generator_params']['input_dropout_p_caption_generation_enc'],
            rnn_cell=config['caption_generator_params']['rnn_type_caption_generation_enc'],
            rnn_dropout_p=config['caption_generator_params']['rnn_dropout_p_caption_generation_enc'],
            bidirectional=config['caption_generator_params']['bidirectional_enc'],
        ),
        DecoderRNN(
            vocab_size=len(vocab),
            max_len=config['caption_generator_params']['max_caption_len'],
            dim_hidden=config['caption_generator_params']['dim_caption_generation_hidden'],
            dim_word=config['caption_generator_params']['dim_word_caption_generation'],
            input_dropout_p=config['caption_generator_params']['input_dropout_p_caption_generation_dec'],
            rnn_cell=config['caption_generator_params']['rnn_type_caption_generation_dec'],
            rnn_dropout_p=config['caption_generator_params']['rnn_dropout_p_caption_generation_dec'],
            bidirectional=config['caption_generator_params']['bidirectional_dec'],
        ),
    )

    retrieval_criterion = ContrastiveLoss(
        margin=config['loss_params']['margin'],
        measure=config['loss_params']['measure'],
        max_violation=config['loss_params']['max_violation'],
        device=device,
    )
    caption_criterion = LanguageModelCriterion()
    # caption_criterion = RewardCriterion()

    model = VSRN(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        caption_model=caption_model,
        retrieval_criterion=retrieval_criterion,
        caption_criterion=caption_criterion,
        weight_retrieval_loss=config['training_params']['weight_retrieval_loss'],
        weight_caption_loss=config['training_params']['weight_caption_loss'],
        learning_rate=config['training_params']['learning_rate'],
        grad_clip=config['training_params']['grad_clip'],
        device=device,
    )

    # load model state
    checkpoint_path = config.get('checkpoint_path')
    use_checkpoint = config.get('use_checkpoint')
    if checkpoint_path and use_checkpoint:
        print(f'loading state dict from checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint['model'])

    return model
