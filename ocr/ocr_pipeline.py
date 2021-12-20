import typing

from ocr.ocr_recognition import Pipeline
from ocr.ocr_settings import WORD_THRESHOLD


def get_tokens_from_images(pipeline: Pipeline, paths: typing.List[str], device):
    result = []
    filtered_result = []
    token_groups, score_groups = pipeline.recognize(paths, device=device)

    for tokens, scores in zip(token_groups, score_groups):
        tokens_for_group = []
        debug_filtered_tokens_group = []
        for token, score in zip(tokens, scores):
            if score >= WORD_THRESHOLD:
                tokens_for_group.append((token[0], score))
            # else:
            #     debug_filtered_tokens_group.append((token[0], score))
        result.append(tokens_for_group) # TODO: Insert dor at the end of sentence
        # filtered_result.append(debug_filtered_tokens_group)
    return result, filtered_result


def get_sentences_from_images(pipeline: Pipeline, paths: typing.List[str], device) -> typing.List[typing.List[str]]:
    """Returns list of sentences (joined all tokens in image) """
    tokens, _ = get_tokens_from_images(pipeline, paths, device)
    sentences_list = [[token[0] for token in group] for group in tokens]
    return sentences_list