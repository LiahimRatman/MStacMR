"""Based on https://github.com/faustomorales/keras-ocr with custom score threshold for recognition"""
import typing

import cv2
import numpy as np
from keras_ocr import tools, detection
from keras_ocr.recognition import PRETRAINED_WEIGHTS, DEFAULT_BUILD_PARAMS, DEFAULT_ALPHABET
from keras_ocr.recognition import build_model
import tensorflow as tf


class Recognizer:
    """A text detector using the CRNN architecture.
    Args:
        alphabet: The alphabet the model should recognize.
        build_params: A dictionary of build parameters for the model.
            See `keras_ocr.recognition.build_model` for details.
        weights: The starting weight configuration for the model.
        include_top: Whether to include the final classification layer in the model (set
            to False to use a custom alphabet).
    """

    def __init__(self, alphabet=None, weights="kurapan", build_params=None):
        assert (
                alphabet or weights
        ), "At least one of alphabet or weights must be provided."
        if weights is not None:
            build_params = build_params or PRETRAINED_WEIGHTS[weights]["build_params"]
            alphabet = alphabet or PRETRAINED_WEIGHTS[weights]["alphabet"]
        build_params = build_params or DEFAULT_BUILD_PARAMS
        if alphabet is None:
            alphabet = DEFAULT_ALPHABET
        self.alphabet = alphabet
        self.blank_label_idx = len(alphabet)
        (
            self.backbone,
            self.model,
            self.training_model,
            self.prediction_model,
        ) = build_model(alphabet=alphabet, **build_params)
        if weights is not None:
            weights_dict = PRETRAINED_WEIGHTS[weights]
            if alphabet == weights_dict["alphabet"]:
                self.model.load_weights(
                    tools.download_and_verify(
                        url=weights_dict["weights"]["top"]["url"],
                        filename=weights_dict["weights"]["top"]["filename"],
                        sha256=weights_dict["weights"]["top"]["sha256"],
                    )
                )
            else:
                print(
                    "Provided alphabet does not match pretrained alphabet. "
                    "Using backbone weights only."
                )
                self.backbone.load_weights(
                    tools.download_and_verify(
                        url=weights_dict["weights"]["notop"]["url"],
                        filename=weights_dict["weights"]["notop"]["filename"],
                        sha256=weights_dict["weights"]["notop"]["sha256"],
                    )
                )

    def get_batch_generator(self, image_generator, batch_size=8, lowercase=False):
        """
        Generate batches of training data from an image generator. The generator
        should yield tuples of (image, sentence) where image contains a single
        line of text and sentence is a string representing the contents of
        the image. If a sample weight is desired, it can be provided as a third
        entry in the tuple, making each tuple an (image, sentence, weight) tuple.
        Args:
            image_generator: An image / sentence tuple generator. The images should
                be in color even if the OCR is setup to handle grayscale as they
                will be converted here.
            batch_size: How many images to generate at a time.
            lowercase: Whether to convert all characters to lowercase before
                encoding.
        """
        y = np.zeros((batch_size, 1))
        if self.training_model is None:
            raise Exception("You must first call create_training_model().")
        max_string_length = self.training_model.input_shape[1][1]
        while True:
            batch = [sample for sample, _ in zip(image_generator, range(batch_size))]
            if not self.model.input_shape[-1] == 3:
                images = [
                    cv2.cvtColor(sample[0], cv2.COLOR_RGB2GRAY)[..., np.newaxis]
                    for sample in batch
                ]
            else:
                images = [sample[0] for sample in batch]
            images = np.array([image.astype("float32") / 255 for image in images])
            sentences = [sample[1].strip() for sample in batch]
            if lowercase:
                sentences = [sentence.lower() for sentence in sentences]
            for c in "".join(sentences):
                assert c in self.alphabet, "Found illegal character: {}".format(c)
            assert all(sentences), "Found a zero length sentence."
            assert all(
                len(sentence) <= max_string_length for sentence in sentences
            ), "A sentence is longer than this model can predict."
            assert all("  " not in sentence for sentence in sentences), (
                "Strings with multiple sequential spaces are not permitted. "
                "See https://github.com/faustomorales/keras-ocr/issues/54"
            )
            label_length = np.array([len(sentence) for sentence in sentences])[
                           :, np.newaxis
                           ]
            labels = np.array(
                [
                    [self.alphabet.index(c) for c in sentence]
                    + [-1] * (max_string_length - len(sentence))
                    for sentence in sentences
                ]
            )
            input_length = np.ones((batch_size, 1)) * max_string_length
            if len(batch[0]) == 3:
                sample_weights = np.array([sample[2] for sample in batch])
                yield (images, labels, input_length, label_length), y, sample_weights
            else:
                yield (images, labels, input_length, label_length), y

    def recognize_from_boxes(
            self, images, box_groups, device, **kwargs
    ) -> typing.Tuple[typing.List[typing.List[str]], typing.List[typing.List[float]]]:
        """Recognize text from images using lists of bounding boxes.
        Args:
            images: A list of input RGB images, supplied as numpy arrays with shape
                (H, W, 3).
            boxes: A list of groups of boxes, one for each image
        """
        assert len(box_groups) == len(
            images
        ), "You must provide the same number of box groups as images."
        crops = []
        start_end: typing.List[typing.Tuple[int, int]] = []
        for image, boxes in zip(images, box_groups):
            image = tools.read(image)
            if self.prediction_model.input_shape[-1] == 1 and image.shape[-1] == 3:
                # Convert color to grayscale
                image = cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)
            for box in boxes:
                crops.append(
                    tools.warpBox(
                        image=image,
                        box=box,
                        target_height=self.model.input_shape[1],
                        target_width=self.model.input_shape[2],
                    )
                )
            start = 0 if not start_end else start_end[-1][1]
            start_end.append((start, start + len(boxes)))
        if not crops:
            return [[]] * len(images), [[]] * len(images)
        X = np.float32(crops) / 255
        if len(X.shape) == 3:
            X = X[..., np.newaxis]

        with tf.device(device):
            all_chars_probs = self.model.predict(X, **kwargs)
            token_scores = np.prod(all_chars_probs.max(-1), -1)
            tokens_predicts = self.prediction_model.predict(X, **kwargs)
        predictions = [
            "".join(
                [
                    self.alphabet[idx]
                    for idx in row
                    if idx not in [self.blank_label_idx, -1]
                ]
            )
            for row in tokens_predicts
        ]
        return [predictions[start:end] for start, end in start_end], \
               [token_scores[start:end].tolist() for start, end in start_end]

    def compile(self, *args, **kwargs):
        """Compile the training model."""
        if "optimizer" not in kwargs:
            kwargs["optimizer"] = "RMSprop"
        if "loss" not in kwargs:
            kwargs["loss"] = lambda _, y_pred: y_pred
        self.training_model.compile(*args, **kwargs)


class Pipeline:
    """A wrapper for a combination of detector and recognizer.
    Args:
        detector: The detector to use
        recognizer: The recognizer to use
        scale: The scale factor to apply to input images
        max_size: The maximum single-side dimension of images for
            inference.
    """

    def __init__(self, detector=None, recognizer=None, scale=2, max_size=2048):
        if detector is None:
            detector = detection.Detector()
        if recognizer is None:
            recognizer = Recognizer()
        self.scale = scale
        self.detector = detector
        self.recognizer = recognizer
        self.max_size = max_size

    def recognize(self, images, detection_kwargs=None, recognition_kwargs=None, device='gpu'):
        """Run the pipeline on one or multiples images.
        Args:
            images: The images to parse (can be a list of actual images or a list of filepaths)
            detection_kwargs: Arguments to pass to the detector call
            recognition_kwargs: Arguments to pass to the recognizer call
        Returns:
            A list of lists of (text, box) tuples.
        """

        # Make sure we have an image array to start with.
        if not isinstance(images, np.ndarray):
            images = [tools.read(image) for image in images]
        # This turns images into (image, scale) tuples temporarily
        images = [
            tools.resize_image(image, max_scale=self.scale, max_size=self.max_size)
            for image in images
        ]
        max_height, max_width = np.array(
            [image.shape[:2] for image, scale in images]
        ).max(axis=0)
        scales = [scale for _, scale in images]
        images = np.array(
            [
                tools.pad(image, width=max_width, height=max_height)
                for image, _ in images
            ]
        )
        if detection_kwargs is None:
            detection_kwargs = {}
        if recognition_kwargs is None:
            recognition_kwargs = {}
        box_groups = self.detector.detect(images=images, **detection_kwargs)
        prediction_groups, pipe_all_scores = self.recognizer.recognize_from_boxes(
            images=images, box_groups=box_groups, device=device, **recognition_kwargs
        )
        box_groups = [
            tools.adjust_boxes(boxes=boxes, boxes_format="boxes", scale=1 / scale)
            if scale != 1
            else boxes
            for boxes, scale in zip(box_groups, scales)
        ]
        return [
                   list(zip(predictions, boxes))
                   for predictions, boxes in zip(prediction_groups, box_groups)
               ], pipe_all_scores
