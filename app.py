import os
import urllib
from pathlib import Path
from typing import List, Dict, Union
import clip
from loguru import logger
import nltk
import numpy as np
import matplotlib.pyplot as plt
import pickle
import streamlit as st
import tensorflow_text # DON'T DELETE IT! (bug in tf)
import tensorflow_hub as tf_hub
import torch

from Vocabulary import Vocabulary
from ocr.ocr_pipeline import get_sentences_from_images
from utilities import get_config, load_from_json
from model import create_model_from_config

from inference_yolov5 import inference_yolo_on_one_image
from inference_clip import inference_clip_one_image
from ocr.ocr_recognition import Pipeline, Recognizer
from ocr.ocr_embedder import OCREmbedder
from yolov5.models.common import DetectMultiBackend


MAX_DETECTIONS_PER_IMAGE = 36
CLIP_EMBEDDING_SIZE = 512


TEMP_DIR = './tmpdir/'
CONFIG_PATH = 'config/full_config_nested.yaml'


def image_load(img_path):
    img = plt.imread(img_path)
    return img


def inference_ocr(image_path: str, storage: Dict):
    ocr_pipeline = storage['ocr']['model']
    muse_embedder = storage['muse']['model']
    sentences = get_sentences_from_images(ocr_pipeline, [image_path], device='cpu')
    sentences = [" ".join(tokens) for tokens in sentences]
    emb_matrix = muse_embedder.get_embeddings_for_gcn(sentences)
    return emb_matrix


def find_nearest_images(caption_embedding, number_of_neighbors: int, space) -> List[str]:
    """return paths to images on disk or with which they might be downloaded"""
    space.set("test", [float(item.item()) for item in caption_embedding[0]])
    neighbors = space.nearest_neighbors(number_of_neighbors, key="test")
    space.multidelete(["test"])

    return neighbors


def find_nearest_captions(image_embedding, number_of_neighbors: int, space) -> List[str]:
    """return nearest captions itself"""
    space.set("test", [float(item.item()) for item in image_embedding[0]])
    neighbors = space.nearest_neighbors(number_of_neighbors, key="test")
    space.multidelete(["test"])

    return neighbors


def inference_on_caption(caption, storage, save_emb=False):
    vsrn_model = storage['vsrn']['model']
    vsrn_vocab = storage['vsrn']['vocab']
    tokens = nltk.tokenize.word_tokenize(str(caption).lower())

    caption = []
    caption.append(vsrn_vocab('<start>'))
    caption.extend([vsrn_vocab(token) for token in tokens])
    caption.append(vsrn_vocab('<end>'))
    caption = torch.Tensor(caption).int()
    with torch.no_grad():
        encoded_caption = vsrn_model.text_encoder(caption.unsqueeze(0), [len(caption)])

    # if save_emb:
    #     save_caption_embedding_to_storage(caption, encoded_caption,hub)

    return encoded_caption


def postprocess_caption(caption: str) -> str:
    return caption


def inference_on_image(image_path, storage, save_emb=False, return_no_ocr_embedding=False):
    model_clip, preprocess_clip = storage['clip']['model'], storage['clip']['preprocess']
    model_yolov5 = storage['yolov5']['model']
    vsrn_model = storage['vsrn']['model']

    detected_regions = inference_yolo_on_one_image(image_path, model_yolov5, torch.device("cpu"))
    region_embeddings = inference_clip_one_image(
        image_path,
        detected_regions,
        model_clip,
        preprocess_clip,
        torch.device("cpu"),
    )
    stacked_image_features = []
    for _ in range(MAX_DETECTIONS_PER_IMAGE):
        if _ < len(region_embeddings):
            stacked_image_features.append(region_embeddings[_])
        else:
            stacked_image_features.append(torch.zeros(CLIP_EMBEDDING_SIZE))
    region_embeddings = np.stack([item.cpu() for item in stacked_image_features], axis=0)

    ocr_embeddings = inference_ocr(image_path, storage)  # [ocr regions(=1) x 512]

    region_embeddings = torch.tensor(region_embeddings).unsqueeze(0)
    ocr_embeddings = torch.tensor(ocr_embeddings).float().unsqueeze(0)
    if torch.cuda.is_available():
        region_embeddings = region_embeddings.cuda()
        ocr_embeddings = ocr_embeddings.cuda()

    # Forward
    with torch.no_grad():
        full_image_embedding, no_ocr_embedding = vsrn_model.image_encoder(region_embeddings, ocr_embeddings)

    # if save_emb:
    #     save_caption_embedding_to_storage(image_path, full_image_embedding, hub)

    if return_no_ocr_embedding:
        return full_image_embedding, no_ocr_embedding

    return full_image_embedding


def inference_generate_caption(image_path, storage, n_top):
    full_image_embedding, no_ocr_embedding = inference_on_image(image_path, storage, return_no_ocr_embedding=True)

    vsrn_model = storage['vsrn']['model']
    vsrn_vocab = storage['vsrn']['vocab']
    generated_captions = []
    for _ in range(n_top):
        seq_logprobs, seq_preds = vsrn_model.caption_model(no_ocr_embedding, None, 'inference')
        # todo попробовать от общего эмбеддинга
        sentence = []
        for letter in seq_preds[0]:
            sentence.append(vsrn_vocab.idx2word[letter.item()])
        generated_captions.append(' '.join(sentence))

    return generated_captions


def get_captions_by_image(image_input: Union[str, Path], n_top: int, storage: Dict, retrieve: bool, ) -> List[str]:
    """
    image_input ideally should be local path to image
    """
    if retrieve:
        image_embedding = inference_on_image(image_input, storage)

        # nearest_captions = find_nearest_captions(image_embedding, n_top, None)
        # nearest_captions = find_nearest_captions(image_embedding, n_top, storage['hub']['caption_space'])

        d = np.dot(image_embedding, storage['caption_embeddings'].T)
        indices = np.zeros(d.shape)
        for i in range(len(indices)):
            indices[i] = np.argsort(d[i])[::-1]
        nearest_captions = [storage['names'][int(ii) // 5]['captions'][int(ii) % 5] for ii in list(indices[0][:n_top])]

    else:
        # image = load_image(image_input)
        nearest_captions = inference_generate_caption(image_input, storage, n_top)
    fetched_captions = [postprocess_caption(caption) for caption in nearest_captions]

    return fetched_captions


def get_images_by_text_query(text_query: str, n_top: int, storage: Dict) -> List[np.array]:
    caption_embedding = inference_on_caption(text_query, storage)

    # nearest_images = find_nearest_images(caption_embedding, n_top, storage['hub']['image_space'])

    d = np.dot(caption_embedding, storage['image_embeddings'].T)
    indices = np.zeros(d.shape)
    for i in range(len(indices)):
        indices[i] = np.argsort(d[i])[::-1]

    base_url = 'http://images.cocodataset.org/train2014/'
    fetched_images = [base_url + storage['names'][int(ii)]['image_path'].split('/')[-1] for ii in
                      list(indices[0][:n_top])]

    # fetched_images = [storage['ctc_map'][image_id]['image_url'] for image_id in nearest_images]

    loaded_images = []
    for i, img_url in enumerate(fetched_images):
        f_name = f'image_example{i}.jpg'
        urllib.request.urlretrieve(img_url, f_name)
        loaded_images.append(image_load(f_name))

    # fetched_images = [image_load('default_image.jpg') for _ in range(n_top)]

    return loaded_images


new_image = False

st.set_page_config(
    page_title='Small, but awesome multimodal search tool demo',
    page_icon=None,
    layout='wide',
    initial_sidebar_state='expanded'
)
st.title('Multimodal Search Demo')


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def instantiate():
    logger.info("Start instantiate all models")
    config = get_config(CONFIG_PATH)

    device = config.get('device', 'cpu')
    vocab = pickle.load(open(config['training_params']['vocab_path'], 'rb'))
    names = load_from_json(config['training_params']['eval_annot_map_path'])
    ctc_map = load_from_json(config["app_params"]["image_map"])
    image_embeddings = np.load((config["app_params"]["image_embeddings"]))
    caption_embeddings = np.load((config["app_params"]["caption_embeddings"]))

    storage = {}
    storage['ctc_map'] = ctc_map
    storage['names'] = names
    storage['caption_embeddings'] = caption_embeddings
    storage['image_embeddings'] = image_embeddings

    # 'clip'    
    storage['clip'] = {}
    clip_model, clip_preprocess = clip.load(config['clip']['model_name'])
    storage['clip']['model'] = clip_model.to(device).eval()
    storage['clip']['preprocess'] = clip_preprocess

    # 'yolov5'
    storage['yolov5'] = {}
    yolov5_model = DetectMultiBackend(config['yolov5']['model_name'], device=device, dnn=False)
    storage['yolov5']['model'] = yolov5_model

    # 'vsrn':
    storage['vsrn'] = {}
    vsrn_model = create_model_from_config(config)
    vsrn_model.eval()
    storage['vsrn']['model'] = vsrn_model
    storage['vsrn']['vocab'] = vocab

    # 'keras-ocr'
    os.environ["KERAS_OCR_CASHE_DIR"] = config["ocr"]["root_pretrained_models"]
    storage['ocr'] = {}
    ocr_model = Pipeline(recognizer=Recognizer())
    storage['ocr']['model'] = ocr_model

    # 'muse'
    storage['muse'] = {}
    muse_path = config['muse']['muse_path']
    embedder = tf_hub.load(muse_path)
    muse_embedder = OCREmbedder(embedder, max_ocr_captions=config['muse']['muse_embedder_max_ocr_captions'])
    storage['muse']['model'] = muse_embedder

    logger.info('All models initialized')

    return storage


def save_image(img, path='saved_image.jpg'):
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    img_path = TEMP_DIR + path
    img_bytes = img.read()
    with open(img_path, 'wb') as f:
        f.write(img_bytes)
    logger.info(f'saving image to {img_path}')

    return img_path


@st.cache(suppress_st_warning=True, ttl=3600, max_entries=1, show_spinner=False)
def load_image(img):
    global new_image
    new_image = True

    if isinstance(img, (str, Path)):
        image = image_load(img)
    elif isinstance(img, np.ndarray):
        return img
    else:
        img_path = save_image(img, 'uploaded_image.jpg')
        # image = image_load(io.BytesIO(img_bytes))
        image = image_load(img_path)
    return image


def main():
    storage = instantiate()

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

    # input processing part
    if IMAGE2TEXT:
        # trying to load image
        is_loaded = False
        img_uploaded = image_uploader_slot.file_uploader(label='Upload your image in .jpg format', type=['jpg', 'jpeg'])
        # img_uploaded is an object, .read() on which returns bytes

        if img_uploaded:
            img = load_image(img_uploaded)
            img_path = save_image(img_uploaded)
            is_loaded = True
            image_slot.image(img, use_column_width=False, width=500)
            if new_image:
                load_status_slot.success('Image loaded!')
            # image_uploader_slot.empty()
    else:
        text_query = caption_slot.text_input(
            label='Type your query to search for images with:',
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

    # fetching part
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
                caption_slot.text(f'\n{"-" * 60}\n'.join(fetched_captions))
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
                with image_slot:
                    fetched_image_slot = st.empty()
                    fetched_image_slot.image(
                        fetched_images, use_column_width=False,
                        width=300 if len(fetched_images) > 1 else 500,
                    )

            else:
                warning_slot.warning('Please, provide the text query you want to search with')

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
