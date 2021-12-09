# import clip
# import nltk
# import numpy as np
# import pickle
# import torch
# import json
# import streamlit as st
# import embeddinghub as eh
#
# from VSRN import VSRN
# from Vocabulary import Vocabulary
# from dao import get_config
# from inference_yolov5 import inference_yolo_on_one_image
# from inference_clip import inference_clip_one_image
# from prepare_image_regions_embeddings import CLIP_EMBEDDING_SIZE, MAX_DETECTIONS_PER_IMAGE
# from models.common import DetectMultiBackend
# from constants import models_for_startup, embeddings_storage_config
#
#
# @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
# def init_preload_model_storage(model_names_list):
#     hub = eh.connect(eh.Config(host="0.0.0.0", port=7462))
#     space = hub.get_space("ctc_image_embs5")
#     with open('CTC_image_name_mapa_new.json', 'r') as f:
#         ctc_map = json.load(f)
#     storage = {}
#     for model_type, item in model_names_list.items():
#         if model_type == 'clip':
#             storage[model_type] = {}
#             clip_model, clip_preprocess = clip.load(item['model_name'])
#             storage[model_type]['model'] = clip_model.to(item['device']).eval()
#             storage[model_type]['preprocess'] = clip_preprocess
#             input_resolution = clip_model.visual.input_resolution
#             context_length = clip_model.context_length
#             vocab_size = clip_model.vocab_size
#             print("CLIP Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
#             print("CLIP Input resolution:", input_resolution)
#             print("CLIP Context length:", context_length)
#             print("CLIP Vocab size:", vocab_size)
#         elif model_type == 'yolov5':
#             storage[model_type] = {}
#             yolov5_model = DetectMultiBackend(item['model_name'], device=item['device'], dnn=False)
#             storage[model_type]['model'] = yolov5_model
#         elif model_type == 'vsrn':
#             storage[model_type] = {}
#             checkpoint = torch.load(item['model_name'], map_location="cpu")
#             vocab = pickle.load(open('checkpoints_and_vocabs/f30k_precomp_vocab.pkl', 'rb'))
#             params = get_config('inference_config.yaml')
#             params['vocab_size'] = len(vocab)
#             vsrn_model = VSRN(params['grad_clip'],
#                               params['image_embedding_dim'],
#                               params['gcn_embedding_size'],
#                               params['vocab_size'],
#                               params['caption_encoder_word_dim'],
#                               params['caption_encoder_num_layers'],
#                               params['caption_encoder_embedding_size'],
#                               params['dim_vid'],
#                               # todo вероятно это то же самое, что и gcn_embedding_size, но надо проверить
#                               params['dim_caption_generation_hidden'],
#                               params['input_dropout_p_caption_generation_enc'],
#                               params['rnn_type_caption_generation_enc'],
#                               params['rnn_dropout_p_caption_generation_enc'],
#                               params['bidirectional_enc'],
#                               params['max_caption_len'],
#                               params['dim_word_caption_generation'],
#                               params['input_dropout_p_caption_generation_dec'],
#                               params['rnn_type_caption_generation_dec'],
#                               params['rnn_dropout_p_caption_generation_dec'],
#                               params['bidirectional_dec'],
#                               params['margin'],
#                               params['measure'],
#                               params['max_violation'],
#                               params['learning_rate'])
#
#             vsrn_model.load_state_dict(checkpoint['model'])
#             vsrn_model.val_start()
#
#             storage[model_type]['model'] = vsrn_model
#             storage[model_type]['vocab'] = vocab
#             storage[model_type]['params'] = params
#
#     return storage, hub, space, ctc_map
#
#
# def inference_on_image(image_path,
#                        storage,
#                        hub,
#                        space,
#                        ctc_map,
#                        save_emb=False):
#     model_clip, preprocess_clip = storage['clip']['model'], storage['clip']['preprocess']
#     model_yolov5 = storage['yolov5']['model']
#     vsrn_model = storage['vsrn']['model']
#     detected_regions = inference_yolo_on_one_image(image_path, model_yolov5, torch.device("cpu"))
#     region_embeddings = inference_clip_one_image(image_path,
#                                                  detected_regions,
#                                                  model_clip,
#                                                  preprocess_clip,
#                                                  torch.device(
#                                                      "cpu"))  # todo Тут могут быть проблемы из-за захардкоженного девайса
#     stacked_image_features = []
#     for _ in range(MAX_DETECTIONS_PER_IMAGE):
#         if _ < len(region_embeddings):
#             stacked_image_features.append(region_embeddings[_])
#         else:
#             stacked_image_features.append(torch.zeros(CLIP_EMBEDDING_SIZE))
#     region_embeddings = np.stack([item.cpu() for item in stacked_image_features], axis=0)
#
#     ocr_embeddings = inference_ocr(region_embeddings)  # Тут должен быть массив размера 16 * 300
#
#     region_embeddings = torch.tensor(region_embeddings).unsqueeze(0)
#     ocr_embeddings = torch.tensor(ocr_embeddings).unsqueeze(0)
#     if torch.cuda.is_available():
#         region_embeddings = region_embeddings.cuda()
#         ocr_embeddings = ocr_embeddings.cuda()
#
#     # Forward
#     with torch.no_grad():
#         full_image_embedding, _ = vsrn_model.img_enc(region_embeddings, ocr_embeddings)
#
#     if save_emb:
#         save_caption_embedding_to_storage(image_path,
#                                           full_image_embedding,
#                                           hub)  # todo Сделать сохранение эмбеддингов
#
#     # if deeper:
#     nearest_caption_ids = find_nearest_caption(full_image_embedding, 3, space, ctc_map)
#
#     return nearest_caption_ids
#
#
# def inference_on_caption(caption,
#                          storage,
#                          hub,
#                          space,
#                          ctc_map,
#                          save_emb=False):
#     vsrn_model = storage['vsrn']['model']
#     vsrn_vocab = storage['vsrn']['vocab']
#     tokens = nltk.tokenize.word_tokenize(str(caption).lower())
#
#     caption = []
#     caption.append(vsrn_vocab('<start>'))
#     caption.extend([vsrn_vocab(token) for token in tokens])
#     caption.append(vsrn_vocab('<end>'))
#     caption = torch.Tensor(caption).int()
#     with torch.no_grad():
#         encoded_caption = vsrn_model.txt_enc(caption.unsqueeze(0), [len(caption)])  # todo Тут бы проверить
#
#     if save_emb:
#         save_caption_embedding_to_storage(caption,
#                                           encoded_caption,
#                                           hub)
#
#     nearest_image_ids = find_nearest_image(encoded_caption, 3, space, ctc_map)
#
#     return nearest_image_ids
#
#
# def inference_generate_caption(image_path,
#                                storage,
#                                save_emb=False):
#     model_clip, preprocess_clip = storage['clip']['model'], storage['clip']['preprocess']
#     model_yolov5 = storage['yolov5']['model']
#     vsrn_model = storage['vsrn']['model']
#     vsrn_vocab = storage['vsrn']['vocab']
#
#     detected_regions = inference_yolo_on_one_image(image_path, model_yolov5, torch.device("cpu"))
#     region_embeddings = inference_clip_one_image(image_path,
#                                                  detected_regions,
#                                                  model_clip,
#                                                  preprocess_clip,
#                                                  torch.device(
#                                                      "cpu"))  # todo Могут быть проблемы из-за захардкоженного девайса
#     stacked_image_features = []
#     for _ in range(MAX_DETECTIONS_PER_IMAGE):
#         if _ < len(region_embeddings):
#             stacked_image_features.append(region_embeddings[_])
#         else:
#             stacked_image_features.append(torch.zeros(CLIP_EMBEDDING_SIZE))
#     region_embeddings = np.stack([item.cpu() for item in stacked_image_features], axis=0)
#     ocr_embeddings = inference_ocr(region_embeddings)
#
#     region_embeddings = torch.tensor(region_embeddings).unsqueeze(0)
#     ocr_embeddings = torch.tensor(ocr_embeddings).unsqueeze(0)
#     if torch.cuda.is_available():
#         region_embeddings = region_embeddings.cuda()
#         ocr_embeddings = ocr_embeddings.cuda()
#
#     # Forward
#     with torch.no_grad():
#         img_emb, GCN_img_emd = vsrn_model.img_enc(region_embeddings, ocr_embeddings)
#
#     seq_logprobs, seq_preds = vsrn_model.caption_model(GCN_img_emd, None, 'inference')
#     # (region_embeddings, ocr_embeddings)  # todo попробовать от общего эмбеддинга
#
#     sentence = []
#     for letter in seq_preds[0]:
#         sentence.append(vsrn_vocab.idx2word[letter.item()])
#
#     generated_caption = ' '.join(sentence)
#
#     return generated_caption
#
#
# def inference_ocr(region_embeddings):  # todo Сделать OCR
#     return np.zeros_like(region_embeddings)[:16, :300]
#
#
# def find_nearest_caption(image_embedding, number_of_neighbors, space, ctc_map):
#     space.set("test", [float(item.item()) for item in image_embedding[0]])
#     neighbors = space.nearest_neighbors(number_of_neighbors, key="test")
#     space.multidelete(["test"])
#
#     return neighbors
#
#
# def find_nearest_image(caption_embedding, number_of_neighbors, space, ctc_map):
#     space.set("test", [float(item.item()) for item in caption_embedding[0]])
#     neighbors = space.nearest_neighbors(number_of_neighbors, key="test")
#     space.multidelete(["test"])
#
#     return neighbors
#
#
# def save_image_embedding_to_storage(image_id,
#                                     image_embedding,
#                                     hub):
#     space = hub.get_space("image_embeddings")
#     space.set(image_id, image_embedding)
#
#
# def save_caption_embedding_to_storage(caption_id,
#                                       caption_embedding,
#                                       hub):
#     space = hub.get_space("caption_embeddings")
#     space.set(caption_id, caption_embedding)
#
#
# def run_api(models_startup):
#     pass
#     # this is a test
#     storage, hub, space, ctc_map = init_preload_model_storage(models_startup)
#     # print(inference_on_image('STACMR_train/CTC/images/COCO_train2014_000000000036.jpg', storage, hub))
#     print(inference_on_caption('Football match', storage, hub, space, ctc_map))
#     # print(inference_generate_caption('STACMR_train/CTC/images/COCO_train2014_000000000036.jpg', storage))
#
#
# run_api(models_for_startup)
#
# # def find_nearest_caption(full_image_embedding, captions_embeddings):
# #     d = np.dot(full_image_embedding, captions_embeddings.T).flatten()
# #     indexes = np.argsort(d)[::-1]
# #
# #     return indexes[0]
# #
# #
# # def find_nearest_image(full_caption_embedding, images_embeddings):
# #     d = np.dot(full_caption_embedding, images_embeddings.T).flatten()
# #     indexes = np.argsort(d)[::-1]
# #
# #     return indexes[0]
#
#
# # def load_image_embeddings_from_storage(image_name,
# #                                        storage_config,
# #                                        storage):
# #     print("Loading image embeddings ...")
# #     image_embs = []
# #     image_paths = [
# #         'STACMR_train/CTC/images/COCO_train2014_000000000036.jpg',
# #         'STACMR_train/CTC/images/COCO_train2014_000000000109.jpg',
# #         'STACMR_train/CTC/images/COCO_train2014_000000000151.jpg',
# #         'STACMR_train/CTC/images/COCO_train2014_000000000260.jpg',
# #         'STACMR_train/CTC/images/COCO_train2014_000000000368.jpg'
# #     ]
# #     for image_path in image_paths:
# #         img_emb = inference_on_image(image_path=image_path,
# #                                      storage=storage,
# #                                      deeper=False)
# #         image_embs.append(img_emb[0])
# #
# #     image_embeddings = np.stack(image_embs, axis=0)
# #
# #     return image_embeddings, image_paths
# #
# #
# # def load_caption_embeddings_from_storage(caption_name,
# #                                          storage_config,
# #                                          storage):  # todo выпилить storage
# #     print("Loading caption embeddings ...")
# #     caption_embs = []
# #     captions_list = [
# #         "A woman posing for the camera, holding a pink, open umbrella and wearing a bright, floral, ruched bathing suit, by a life guard stand with lake, green trees, and a blue sky with a few clouds behind.",
# #         "Very bad day is today",
# #         "Dogs are my best friends",
# #         "I love to eat wood",
# #         "Such a wonderful night"
# #     ]
# #     for caption in captions_list:
# #         cap_emb = inference_on_caption(caption=caption,
# #                                        storage=storage,
# #                                        deeper=False)
# #         caption_embs.append(cap_emb[0])
# #
# #     captions_embeddings = np.stack(caption_embs, axis=0)
# #
# #     return captions_embeddings, captions_list
