import streamlit as st
import os
from matplotlib import pyplot as plt
import torch
from inference_service import models_for_startup, init_preload_model_storage, inference_generate_caption
from Vocabulary import Vocabulary


### PATHS
UTILITIES_PATH = './utilities/'
DATA_PATH = './data_my/'

### PATHS TO MODELS
# CAPTION_MODEL_PATH = UTILITIES_PATH + 'caption_net.pth'
# CAPTION_MODEL_GDRIVE_ID = '1fgnUGsS1bl_lZk_cZP-FjJ1GwS5Sr4A7'
# INCEPTION_PATH = UTILITIES_PATH + 'bh_inception.pth'
# INCEPTION_GDRIVE_ID = '1ByN-JYRPyOYUXwvmhzz54-yfz8-aTpNC'

TMP_DIR = 'tmp/'
TMP_IMG_PATH = TMP_DIR + 'img.jpg'


@st.cache(suppress_st_warning=True, ttl=3600, max_entries=1, show_spinner=False)
def load_image(img):
    if isinstance(img, str):
        return img
    else:
        img_bytes = img.read()
        if not os.path.exists(TMP_DIR):
            os.mkdir(TMP_DIR)
        with open(TMP_IMG_PATH, 'wb') as f:
            f.write(img_bytes)
        return TMP_IMG_PATH


def main():
    storage, hub, space, ctc_map = init_preload_model_storage(models_for_startup)

    spinner_slot = st.empty()
    left, right = st.beta_columns((1, 1))
    image_slot = left
    caption_slot = right

    is_loaded = False
    img_loaded = st.file_uploader(label='Upload your image in .jpg format', type=['jpg', 'jpeg', 'png'])
    load_status_slot = st.empty()

    if img_loaded:
        img_path = load_image(img_loaded)
        is_loaded = True
        # img, img_for_net, img_size_init = image_process(img)
        image = plt.imread(img_path)
        image_slot.image(image, use_column_width=False, width=600)
        # if new_image:
        #     load_status_slot.success('Image loaded!')

    # MAX_CAPTION_LEN = st.sidebar.number_input("Select maximal caption length:",
    #                                           min_value=1, max_value=20,
    #                                           value=8, step=1)
    # SAMPLING = st.sidebar.radio("Should sampling be done?", ('Sampling', 'Use Best Option'))
    # SAMPLE = SAMPLING == 'Sampling'
    #
    # n_captions_slot = st.sidebar.empty()
    # slider_slot = st.sidebar.empty()
    #
    # if SAMPLE:
    #     N_CAPTIONS = n_captions_slot.number_input("Select number of captions generated:",
    #                                               min_value=1, max_value=15, value=5, step=1)
    #     TEMPERATURE = slider_slot.slider("Set temperature for sampling: ",
    #                                      min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    # else:
    #     N_CAPTIONS = 1
    #     TEMPERATURE = 1
    #     slider_slot.markdown('No sampling would be made while generating the one and only caption variant')

    button_slot = st.sidebar.empty()
    warning_slot = st.sidebar.empty()
    authors_slot = st.sidebar.empty()

    if button_slot.button('Generate Captions!'):
        load_status_slot.empty()
        if is_loaded:
            spinner_slot.info('Generating...')
            generated_caption = inference_generate_caption(img_path, storage)
            # generated_captions = get_captions(img_for_net, inception, caption_model, idx2word,
            #                                   EXCLUDE_FROM_PREDICTION, (BOS_IDX,), EOS_IDX,
            #                                   N_CAPTIONS, TEMPERATURE, SAMPLE, MAX_CAPTION_LEN)
            spinner_slot.empty()
            caption_slot.header('What are we seeing there..')
            caption_slot.markdown(generated_caption)
            # caption_slot.markdown('  \n'.join(generated_captions))
        else:
            warning_slot.warning('Please, upload your image first')

    authors_slot.markdown("""\
    <span style="color:black;font-size:8"><p>\
    made by\
    <a style="color:green" href="https://data.mail.ru/profile/a.nalitkin/">michael</a>
    &
    <a style="color:crimson" href="https://data.mail.ru/profile/m.korotkov/">aleksandr</a>
    &
    <a style="color:magenta" href="https://data.mail.ru/profile/m.zavgorodnyaya/">marina</a>
    </p></span>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
