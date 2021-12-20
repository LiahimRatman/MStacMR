from typing import List

import numpy as np


class OCREmbedder:
    def __init__(self, embedder, max_ocr_captions=256):
        self.embedder = embedder
        self.emb_dim = len(self.get_single_sentence_embedding('test'))
        self.max_ocr_captions = max_ocr_captions

    def get_single_sentence_embedding(self, sentence):
        assert isinstance(sentence, str)
        return self.embedder(sentence).numpy()[0]

    def get_embeddings_for_gcn(self, scene_texts: List[str]):
        """
        every element in the list should be full ocr-recognized text, tokenization happens inside
        """
        embeddings = np.zeros((self.max_ocr_captions, self.emb_dim))
        stopper = min(self.max_ocr_captions, len(scene_texts))
        for ix, text in enumerate(scene_texts[:stopper]):
            embeddings[ix, :] = self.get_single_sentence_embedding(text)
        return embeddings