import pickle

from utilities import get_config
from data_utils import get_dataloader_img_ocr_precalculated
from evaluation_utils import evaluate
from Vocabulary import Vocabulary

from model import create_model_from_config

INFERENCE_CONFIG_PATH = 'config/full_config_nested.yaml'


def run_evaluation(config):
    model = create_model_from_config(config)

    vocab = pickle.load(open(config['training_params']['vocab_path'], 'rb'))

    val_loader = get_dataloader_img_ocr_precalculated(
        annotation_map_path=config['training_params']['eval_annot_map_path'],
        image_embeddings_path=config['training_params']['eval_img_emb_path'],
        ocr_embeddings_path=config['training_params']['eval_ocr_emb_path'],
        encoder_tokenizer_path=config['caption_encoder_params']['encoder_path'],
        vocab=vocab,
        shuffle=False,
        batch_size=config['training_params']['batch_size'],
        num_image_boxes=config['image_encoder_params']['num_image_boxes'],
    )

    similarity_measure = config['loss_params']['measure']
    current_score = evaluate(model, val_loader, similarity_measure)

    return current_score


if __name__ == '__main__':
    config = get_config(INFERENCE_CONFIG_PATH)
    run_evaluation(config)
