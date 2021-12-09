import sys
import os
import io
import gc
import warnings
import pickle
import streamlit as st
import numpy as np
from pathlib import Path
from typing import List, Set, Dict, Union
import clip
import nltk
import numpy as np
import pickle
import torch
import json
import streamlit as st
import embeddinghub as eh

from VSRN import VSRN
from Vocabulary import Vocabulary
from dao import get_config
from inference_yolov5 import inference_yolo_on_one_image
from inference_clip import inference_clip_one_image
from prepare_image_regions_embeddings import CLIP_EMBEDDING_SIZE, MAX_DETECTIONS_PER_IMAGE
from models.common import DetectMultiBackend
from constants import models_for_startup, embeddings_storage_config

from caption_utils import download_file_from_google_drive, load_models, image_load, image_process, get_captions

### PATHS
UTILITIES_PATH = './utilities/'
DATA_PATH = './data/'

### PATHS TO MODELS
CAPTION_MODEL_PATH = UTILITIES_PATH + 'caption_net.pth'
CAPTION_MODEL_GDRIVE_ID = '1fgnUGsS1bl_lZk_cZP-FjJ1GwS5Sr4A7'
INCEPTION_PATH = UTILITIES_PATH + 'bh_inception.pth'
INCEPTION_GDRIVE_ID = '1ByN-JYRPyOYUXwvmhzz54-yfz8-aTpNC'

TMP_DIR = UTILITIES_PATH + '/tmp/'
TMP_IMG_PATH = TMP_DIR + 'img.jpg'


# def find_nearest_images(caption_embedding, number_of_neighbors: int, hub = None, ) -> List[str]:
# """return paths to images on disk or with which they might be downloaded"""
# space = hub.get_space("image_embeddings")
# neighbors = space.nearest_neighbors(number_of_neighbors, vector=caption_embedding)
# return neighbors

# def find_nearest_captions(image_embedding, number_of_neighbors: int, hub = None) -> List[str]:
# """return nearest captions itself"""
# space = hub.get_space("caption_embeddings")
# neighbors = space.nearest_neighbors(number_of_neighbors, vector=image_embedding)
# return neighbors

def find_nearest_images(caption_embedding, number_of_neighbors: int, hub=None, ) -> List[str]:
    """return paths to images on disk or with which they might be downloaded"""
    space = hub.get_space("image_embeddings")
    neighbors = space.nearest_neighbors(number_of_neighbors, vector=caption_embedding)
    return neighbors


def find_nearest_captions(image_embedding, number_of_neighbors: int, hub=None) -> List[str]:
    """return nearest captions itself"""
    space = hub.get_space("caption_embeddings")
    neighbors = space.nearest_neighbors(number_of_neighbors, vector=image_embedding)
    return neighbors


def inference_on_caption(caption_text: str, storage: Dict) -> np.array:
    return np.zeros(10)


def inference_on_image(image: np.array, storage: Dict) -> np.array:
    return np.zeros(10)


def full_inference_on_image(image: np.array, n_top: int, storage: Dict) -> List[str]:
    image, img_for_net, image_size_init = image_process(image)

    captions = get_captions(
        img_for_net,
        storage['inception'], storage['caption_model'],
        storage['idx2word'],
        storage['EXCLUDE_FROM_PREDICTION'], (storage['BOS_IDX'],), storage['EOS_IDX'],
        n_top,
        storage['TEMPERATURE'], storage['SAMPLE'], storage['MAX_CAPTION_LEN']
    )

    return captions


def get_images_by_text_query(text_query: str, n_top: int, storage: Dict) -> List[np.array]:
    # fetched_images = [f'fetched #{i + 1}' for i in range(N_TOP_RESULTS)]

    # caption_embedding = inference_on_caption(text_query, storage)
    # nearest_images = find_nearest_images(caption_embedding, n_top)
    # fetched_images = [image_load(img_path) for img_path in nearest_images]

    fetched_images = [image_load('default_image.jpg') for _ in range(n_top)]

    return fetched_images


def postprocess_caption(caption: str) -> str:
    return caption


def get_captions_by_image(image_input: Union[str, Path], n_top: int, storage: Dict, retrieve: bool, ) -> List[str]:
    """
    image_input ideally should be local path to image
    """
    if retrieve:
        image_embedding = inference_on_image(image, storage)
        nearest_captions = find_nearest_captions(image_embedding, n_top)
    else:
        image = load_image(image_input)
        nearest_captions = full_inference_on_image(image, n_top, storage)
    fetched_captions = [postprocess_caption(caption) for caption in nearest_captions]

    return fetched_captions


new_image = False

st.set_page_config(
    page_title='Small, but awesome image captioning tool demo',
    page_icon=None,
    layout='wide',
    initial_sidebar_state='expanded'
)
st.title('Multimodal Search Demo')


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def init_preload_model_storage(model_names_list):
    hub = eh.connect(eh.Config(host="0.0.0.0", port=7462))
    space = hub.get_space("ctc_image_embs5")
    with open('CTC_image_name_mapa_new.json', 'r') as f:
        ctc_map = json.load(f)
    storage = {}
    for model_type, item in model_names_list.items():
        if model_type == 'clip':
            storage[model_type] = {}
            clip_model, clip_preprocess = clip.load(item['model_name'])
            storage[model_type]['model'] = clip_model.to(item['device']).eval()
            storage[model_type]['preprocess'] = clip_preprocess
            input_resolution = clip_model.visual.input_resolution
            context_length = clip_model.context_length
            vocab_size = clip_model.vocab_size
            print("CLIP Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
            print("CLIP Input resolution:", input_resolution)
            print("CLIP Context length:", context_length)
            print("CLIP Vocab size:", vocab_size)
        elif model_type == 'yolov5':
            storage[model_type] = {}
            yolov5_model = DetectMultiBackend(item['model_name'], device=item['device'], dnn=False)
            storage[model_type]['model'] = yolov5_model
        elif model_type == 'vsrn':
            storage[model_type] = {}
            checkpoint = torch.load(item['model_name'], map_location="cpu")
            vocab = pickle.load(open('checkpoints_and_vocabs/f30k_precomp_vocab.pkl', 'rb'))
            params = get_config('inference_config.yaml')
            params['vocab_size'] = len(vocab)
            vsrn_model = VSRN(params['grad_clip'],
                              params['image_embedding_dim'],
                              params['gcn_embedding_size'],
                              params['vocab_size'],
                              params['caption_encoder_word_dim'],
                              params['caption_encoder_num_layers'],
                              params['caption_encoder_embedding_size'],
                              params['dim_vid'],
                              # todo вероятно это то же самое, что и gcn_embedding_size, но надо проверить
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

            vsrn_model.load_state_dict(checkpoint['model'])
            vsrn_model.val_start()

            storage[model_type]['model'] = vsrn_model
            storage[model_type]['vocab'] = vocab
            storage[model_type]['params'] = params

    return storage, hub, space, ctc_map


@st.cache(suppress_st_warning=True, ttl=3600, max_entries=1, show_spinner=False)
def load_image(img):
    global new_image
    new_image = True

    if isinstance(img, (str, Path)):
        image = image_load(img)
    elif isinstance(img, np.ndarray):
        return img
    else:
        img_bytes = img.read()
        if not os.path.exists(TMP_DIR):
            os.mkdir(TMP_DIR)
        with open(TMP_IMG_PATH, 'wb') as f:
            f.write(img_bytes)
        # image = image_load(io.BytesIO(img_bytes))
        image = image_load(TMP_IMG_PATH)
    return image


def save_image(img):
    if not os.path.exists(TMP_DIR):
        os.mkdir(TMP_DIR)
    img_path = TMP_DIR + 'saved_image.jpg'

    img_bytes = img.read()
    with open(img_path, 'wb') as f:
        f.write(img_bytes)

    return img_path


def main():
    download, pseudo_inception = True, False
    #
    # storage = instantiate(download, pseudo_inception)
    storage, hub, space, ctc_map = init_preload_model_storage(models_startup)

    spinner_slot = st.empty()
    load_status_slot = st.empty()

    left, right = st.beta_columns((1, 1))
    with left:
        image_slot = st.empty()
        image_uploader_slot = st.empty()

    caption_slot = right

    SEARCH_DIRECTION = st.sidebar.radio(
        "From which to which modality should we search?",
        ('Image2Text', 'Text2Image')
    )
    IMAGE2TEXT = (SEARCH_DIRECTION == 'Image2Text')

    n_top_results_slot = st.sidebar.empty()
    N_TOP_RESULTS = n_top_results_slot.number_input(
        "How many relevant results you want to get:",
        min_value=1, max_value=10, value=5, step=1
    )

    ### input processing part
    if IMAGE2TEXT:
        ### trying to load image
        is_loaded = False
        img_uploaded = image_uploader_slot.file_uploader(label='Upload your image in .jpg format', type=['jpg', 'jpeg'])
        ### img_uploaded is an object .read() on which returns bytes

        if img_uploaded:
            img = load_image(img_uploaded)
            img_path = save_image(img_uploaded)
            is_loaded = True
            image_slot.image(img, use_column_width=False, width=600)
            if new_image:
                load_status_slot.success('Image loaded!')
            # image_uploader_slot.empty()
    else:
        text_query = caption_slot.text_input(
            label='Type your query to search for image with:',
            value="", max_chars=None,
            key=None, type="default",
        )

        if text_query:
            image_slot.markdown(text_query)
            load_status_slot.success('Your prompt input has been saved!')

    retrieval_slot = st.sidebar.empty()
    caption_length_slot = st.sidebar.empty()
    temperature_slot = st.sidebar.empty()

    button_slot = st.sidebar.empty()
    warning_slot = st.sidebar.empty()
    authors_slot = st.sidebar.empty()

    ### fetching part
    if IMAGE2TEXT:
        CAPTION_CREATION = retrieval_slot.radio(
            "Retrieve caption from existing or generate from scratch?",
            ('Retrieve', 'Generate')
        )
        RETRIEVE = (CAPTION_CREATION == 'Retrieve')
        if RETRIEVE:
            SAMPLE = False
            MAX_CAPTION_LEN = 10
            TEMPERATURE = 0.8
        else:
            SAMPLE = True
            MAX_CAPTION_LEN = caption_length_slot.number_input(
                "Set maximal caption length:",
                min_value=1, max_value=20, value=8, step=1,
            )
            TEMPERATURE = temperature_slot.slider(
                "Set temperature for sampling: ",
                min_value=0.1, max_value=2.0, value=0.5, step=0.1,
            )

        storage['MAX_CAPTION_LEN'] = MAX_CAPTION_LEN
        storage['TEMPERATURE'] = TEMPERATURE
        storage['SAMPLE'] = SAMPLE

        if button_slot.button('Fetch!'):
            load_status_slot.empty()
            if is_loaded:
                spinner_slot.info('Fetching...')
                fetched_captions = get_captions_by_image(
                    img_path,
                    n_top=N_TOP_RESULTS,
                    storage=storage,
                    retrieve=RETRIEVE,
                )
                spinner_slot.empty()
                caption_slot.header('Is this what you were searching for?')
                caption_slot.markdown('  \n'.join(fetched_captions))
            else:
                warning_slot.warning('Please, upload your image first')

    else:
        if button_slot.button('Fetch!'):
            load_status_slot.empty()
            if text_query:
                spinner_slot.info('Fetching...')

                fetched_images = get_images_by_text_query(
                    text_query=text_query,
                    n_top=N_TOP_RESULTS,
                    storage=storage,
                )
                spinner_slot.empty()
                image_slot.header('Is this what you were searching for?')
                # image_slot.image(fetched_images[0]) ### only first one for now
                with image_slot:
                    fetched_image_slot = st.empty()
                    fetched_image_slot.image(
                        fetched_images, use_column_width=False,
                        width=300 if len(fetched_images) > 1 else 600,
                    )

            else:
                warning_slot.warning('Please, provide the text query you want to search')

    authors_slot.markdown(
        """\
        <span style="color:black;font-size:8"><p>\
        made by\
        \n
        <a style="color:mediumorchid" href="https://data.mail.ru/profile/a.nalitkin/">aleksandr</a>
        \n
        <a style="color:mediumorchid" href="https://data.mail.ru/profile/m.korotkov/">michael</a>
        \n
        <a style="color:mediumorchid" href="https://data.mail.ru/profile/m.zavgorodnyaya/">marina</a>
        </p></span>
        """,
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()
