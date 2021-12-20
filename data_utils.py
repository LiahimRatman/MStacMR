import nltk
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModel
from transformers import BatchEncoding

from collections import defaultdict

from utilities import load_from_json


def get_tokens_from_encoded_inputs(encoded_input, tokenizer):

    separator = ' '
    spec_token_ids = set([i for i in [
        tokenizer.pad_token_id,
        tokenizer.mask_token_id,
        tokenizer.bos_token_id, 
        tokenizer.eos_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.unk_token_id,
    ] if i is not None])

    input_ids = encoded_input['input_ids'].cpu().numpy() ### batch x seq_len
    output = []
    for seq in input_ids:
        clear_seq = [id for id in seq if id not in spec_token_ids]
        tokens = tokenizer.convert_ids_to_tokens(clear_seq)
        sentence = tokenizer.convert_tokens_to_string(tokens).split(separator)
        # sentence = tokens
        output.append(sentence)

    return output



def prepare_captions(caption, vocab, max_len):
    # todo make here a function for caption encoding
    
    ### assumed padding is 0
    # Convert caption (string) to word ids.
    tokens = nltk.tokenize.word_tokenize(str(caption).lower())

    indices = []
    indices.append(vocab('<start>'))
    indices.extend([vocab(token) for token in tokens])
    indices.append(vocab('<end>'))
    caption_token_ids = torch.Tensor(indices)

    # Deal with caption model data
    # label = np.zeros(self.max_len)
    mask = np.zeros(max_len + 1)
    gts = np.zeros((max_len + 1))

    # print(tokens)
    cap_caption = ['<start>'] + tokens + ['<end>']
    # print(cap_caption)
    if len(cap_caption) > max_len - 1:
        cap_caption = cap_caption[:max_len]
        cap_caption[-1] = '<end>'

    for j, w in enumerate(cap_caption):
        gts[j] = vocab(w)

    non_zero = (gts == 0).nonzero()

    mask[:int(non_zero[0][0]) + 1] = 1

    caption_label = torch.from_numpy(gts).type(torch.LongTensor)
    caption_mask = torch.from_numpy(mask).type(torch.FloatTensor)

    ### full length seq, cut or padded to max_len, cut or padded to max_len,
    return caption_token_ids, caption_label, caption_mask


class FullDataset:
    def __init__(
        self,
        annotation_map_path=None,
        image_embeddings_path=None,
        ocr_embeddings_path=None,
        encoder_tokenizer_path=None,
        vocab=None,
        captions_per_image=5,
        max_ocr_regions=16,
        max_caption_len=100,
        use_precomputed_embeddings=True,
    ):
        self.annotation_map_path = annotation_map_path
        self.image_embeddings_path = image_embeddings_path
        self.ocr_embeddings_path = ocr_embeddings_path
        self.encoder_tokenizer_path = encoder_tokenizer_path
        self.use_precomputed_embeddings = use_precomputed_embeddings
        self.captions_per_image = captions_per_image
        self.max_ocr_regions = max_ocr_regions
        self.max_caption_len = max_caption_len
        self.vocab = vocab

        self.encoder_tokenizer = AutoTokenizer.from_pretrained(self.encoder_tokenizer_path)
        
        if self.use_precomputed_embeddings:
            self.precomputed_image_embeddings = np.load(self.image_embeddings_path)
            # ocr_dummy = np.zeros(
            #     (self.precomputed_image_embeddings.shape[0], max_ocr_regions, 300),
            #     dtype='float16'
            # )
            # self.precomputed_ocr_embeddings = ocr_dummy
            self.precomputed_ocr_embeddings = np.load(self.ocr_embeddings_path)

        self.annotation_map = load_from_json(self.annotation_map_path)
        assert len(self.precomputed_image_embeddings) == len(self.annotation_map)
        self.n_items = len(self.precomputed_image_embeddings)
        self.length = self.n_items * self.captions_per_image
        #### every item holds self.captions_per_image text_captions and 1 path to image

    def get_caption_from_index(self, index):
        ### indices 0:(self.captions_per_image-1) holds actually hold the same image but different captions for it
        ### same for self.captions_per_image:(2*self.captions_per_image-1)
        item_id = index // self.captions_per_image
        caption_idx = index % self.captions_per_image
        item = self.annotation_map[item_id]
        caption = item['captions'][caption_idx]
        return caption

    def __getitem__(self, index):
        
        item_id = index // self.captions_per_image
        caption = self.get_caption_from_index(index) ### str
        
        caption_token_ids, caption_label, caption_mask = prepare_captions(
            caption=caption,
            vocab=self.vocab,
            max_len=self.max_caption_len
        )

        # tokenizer_outputs = self.encoder_tokenizer(
        #     caption,
        #     max_length = self.max_caption_len,
        #     padding = 'max_length',
        #     truncation = True,
        #     return_tensors='pt',
        # )
        tokenizer_outputs = caption
        # print(type(tokenizer_outputs))

        if self.use_precomputed_embeddings:
            image_regions_embedding = self.precomputed_image_embeddings[item_id]
            image_regions_embedding = torch.Tensor(image_regions_embedding)

            ocr_embedding = self.precomputed_ocr_embeddings[item_id]
            # ограничим максимальное количество элементов текста сцены. Можно сделать аналогично и для картинок при желании
            ocr_embedding = ocr_embedding[:self.max_ocr_regions][:]  # todo в чем смысл последнего двоеточия..
            ocr_embedding = torch.Tensor(ocr_embedding)
            
        else:
            raise NotImplementedError('design full train pipeline here. At this moment is not created')
            image_path = item['image_path']

        return (
            index, ### full index in dataset - with respect to amount of captions per image
            image_regions_embedding, ### n_bboxes * bb_emb_dim
            ocr_embedding, ### ???? i guess max_ocr_regions * ocr_emb_dim ????
            # tokenizer_outputs, ### dict with 'input_ids' tensor 1 x max_len, 'token_type_ids', 'attention_mask' of the same size 
            tokenizer_outputs, ### str
            caption_token_ids, ### init_caption_tokens_ids as tensor
            caption_label, ### same as caption_vocab_indices, but cut or padded to max_len
            caption_mask, ### same as caption_vocab_indices, but cut or padded to max_len
        )

    def __len__(self):
        return self.length


def precomputed_collate_fn(data):
    """Build mini-batch tensors from a list of tuples."""
    
    # sorting list by actual captions length

    CAPTION_INDEX_IN_TUPLE = 4

    data.sort(key=lambda x: len(x[CAPTION_INDEX_IN_TUPLE]), reverse=True)
    (
        ids,
        image_embeddings,
        ocr_embeddings,
        tokenizer_outputs,
        caption_token_ids,
        caption_labels,
        caption_masks
     ) = zip(*data)

    image_embeddings = torch.stack(image_embeddings, 0)
    ocr_embeddings = torch.stack(ocr_embeddings, 0)
    
    # stacked = defaultdict(list)
    # [stacked[k].append(v) for d in tokenizer_outputs for k, v in d.items()]
    # tokenizer_outputs = BatchEncoding({k: torch.cat(v, 0) for k, v in stacked.items()})
    

    # print(type(tokenizer_outputs))

    caption_labels = torch.stack(caption_labels, 0)
    caption_masks = torch.stack(caption_masks, 0)

    # pad captions to actual max length in batch
    lengths = [len(cap) for cap in caption_token_ids]
    targets = torch.zeros(len(caption_token_ids), max(lengths)).long()
    for i, cap in enumerate(caption_token_ids):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return ids, image_embeddings, ocr_embeddings, tokenizer_outputs, targets, lengths, caption_labels, caption_masks


def get_dataloader_img_ocr_precalculated(
    annotation_map_path,
    image_embeddings_path,
    ocr_embeddings_path,
    encoder_tokenizer_path,
    vocab,
    shuffle,
    batch_size = 128,
    num_workers = 0,
    max_caption_len = 100,
    ):

    dataset = FullDataset(
        annotation_map_path=annotation_map_path,
        image_embeddings_path=image_embeddings_path,
        ocr_embeddings_path=ocr_embeddings_path,
        encoder_tokenizer_path=encoder_tokenizer_path,
        vocab=vocab,
        max_caption_len = max_caption_len,
        use_precomputed_embeddings=True,        
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,  ### check this one
        num_workers=num_workers,
        collate_fn = precomputed_collate_fn,
    )

    return dataloader


class InferenceImagesEncodeDataset:
    def __init__(self, inference_images_path):
        self.inference_images_path = inference_images_path

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class InferenceCaptionsEncodeDataset:
    def __init__(self, inference_texts_path):
        self.inference_texts_path = inference_texts_path

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
