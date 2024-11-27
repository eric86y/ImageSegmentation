import os
import cv2
import json
import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from Source.Utils import optimize_countour

class LayoutDetection:
    """
    Handles layout detection
    Args:
        - config_file: json file with the following parameters:
            - onnx model file
            - image input width and height
            - classes

    """

    def __init__(
        self,
        config_file: str,
    ) -> None:
        self._config_file = config_file
        self._onnx_model_file = None
        self._input_width = 1024
        self._input_height = 320
        self._class_dict = None
        self._can_run = False
        self._inference = None
        # add other Execution Providers if applicable, see: https://onnxruntime.ai/docs/execution-providers
        self.execution_providers = ["CPUExecutionProvider", "CUDAExecutionProvider"]

        self._init()

    def _init(self) -> None:
        _file = open(self._config_file)
        json_content = json.loads(_file.read())
        self._onnx_model_file = json_content["model"]
        self._input_width = json_content["input_width"]
        self._input_height = json_content["input_height"]
        self._class_dict = json_content["classes"]

        if self._onnx_model_file is not None:
            model_file_path = f"{os.path.dirname(self._config_file)}/{self._onnx_model_file}"
            try:
                self._inference = ort.InferenceSession(
                    model_file_path, providers=self.execution_providers
                )
                self.can_run = True

            except Exception as error:
                print.error(f"Error loading model file: {error}")
                self.can_run = False
        else:
            self.can_run = False

        print(f"Layout Inference -> Init(): {self.can_run}")

    def prepare_img_patches(self, image: np.array) -> np.array:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)
        image /= 255.0
        image = np.dstack([image, image, image])
        return image

    def get_contours(
        self, prediction: np.array, optimize: bool = True, return_bbox: bool = False
    ) -> list:
        prediction = np.where(prediction > 200, 255, 0)
        prediction = prediction.astype(np.uint8)

        if np.sum(prediction) > 0:
            contours, _ = cv2.findContours(
                prediction, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            if return_bbox:
                # TODO: implement this function
                pass

            elif optimize:
                contours = [optimize_countour(x) for x in contours]

            else:
                return contours

        else:
            return []

    def generate_page_data(
        self, original_image: np.array, predictions: np.array, alpha: float = 0.4
    ) -> np.array:
        
        predictions = cv2.resize(
                predictions, (original_image.shape[1], original_image.shape[0])
            )

        pred_images = predictions[:, :, 0]
        pred_lines = predictions[:, :, 1]
        pred_margin = predictions[:, :, 2]
        pred_caption = predictions[:, :, 3]

        preview_img = np.zeros(
            shape=(predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8
        )

        if np.sum(pred_lines) > 0:
            contours, _ = cv2.findContours(
                pred_lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            color = tuple([int(x) for x in self._class_dict["line"].split(",")])

            for idx, _ in enumerate(contours):
                cv2.drawContours(
                    preview_img, contours, contourIdx=idx, color=color, thickness=-1
                )

        if np.sum(pred_images) > 0:
            contours, _ = cv2.findContours(
                pred_images, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            color = tuple([int(x) for x in self._class_dict["image"].split(",")])

            for idx, _ in enumerate(contours):
                cv2.drawContours(
                    preview_img, contours, contourIdx=idx, color=color, thickness=-1
                )

        if np.sum(pred_margin) > 0:
            contours, _ = cv2.findContours(
                pred_margin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            color = tuple([int(x) for x in self._class_dict["margin"].split(",")])

            for idx, _ in enumerate(contours):
                cv2.drawContours(
                    preview_img, contours, contourIdx=idx, color=color, thickness=-1
                )

        if np.sum(pred_caption) > 0:
            contours, _ = cv2.findContours(
                pred_caption, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            color = tuple([int(x) for x in self._class_dict["caption"].split(",")])

            for idx, _ in enumerate(contours):
                cv2.drawContours(
                    preview_img, contours, contourIdx=idx, color=color, thickness=-1
                )

        preview_img = cv2.resize(
            preview_img,
            (original_image.shape[1], original_image.shape[0]),
        )

        cv2.addWeighted(
            preview_img, alpha, original_image, 1 - alpha, 0, original_image
        )

        return original_image

    def create_preview_image(
        self,
        image: np.array,
        image_predictions: list,
        line_predictions: list,
        caption_predictions: list,
        margin_predictions: list,
        alpha: float = 0.4,
    ) -> np.array:
        # preview_image = image.copy() # huh?
        mask = np.zeros(image.shape, dtype=np.uint8)

        if len(image_predictions) > 0:
            color = tuple([int(x) for x in self._class_dict["image"].split(",")])

            for idx, _ in enumerate(image_predictions):
                cv2.drawContours(
                    mask, image_predictions, contourIdx=idx, color=color, thickness=-1
                )

        if len(line_predictions) > 0:
            color = tuple([int(x) for x in self._class_dict["line"].split(",")])

            for idx, _ in enumerate(line_predictions):
                cv2.drawContours(
                    mask, line_predictions, contourIdx=idx, color=color, thickness=-1
                )

        if len(caption_predictions) > 0:
            color = tuple([int(x) for x in self._class_dict["caption"].split(",")])

            for idx, _ in enumerate(caption_predictions):
                cv2.drawContours(
                    mask, caption_predictions, contourIdx=idx, color=color, thickness=-1
                )

        if len(margin_predictions) > 0:
            color = tuple([int(x) for x in self._class_dict["margin"].split(",")])

            for idx, _ in enumerate(margin_predictions):
                cv2.drawContours(
                    mask, margin_predictions, contourIdx=idx, color=color, thickness=-1
                )

        cv2.addWeighted(mask, alpha, image, 1 - alpha, 0, image)

        return image

    def _predict(self, img_patches: np.array, class_threshold: float) -> np.array:
        img_batch = [self.prepare_img_patches(x) for x in img_patches]
        img_batch = np.array(img_batch)
        img_bach = np.transpose(img_batch, axes=[0, 3, 1, 2])
        #img_batch = np.expand_dims(img_batch, axis=1)
        print(f"Input: {img_batch.shape}")
        img_bach = ort.OrtValue.ortvalue_from_numpy(img_bach)
        pred_batch = self._inference.run(None, {"input": img_batch})
        pred_batch = pred_batch[0].numpy()
        print(f"Predictions: {pred_batch.shape}")
        predictions = np.squeeze(predictions[0], axis=0)
        predictions = softmax(predictions, axis=0)
        predictions = np.transpose(predictions, axes=[1, 2, 0])

        predictions = predictions[:, :, 1:]  # removes the background class
        predictions = np.where(predictions > class_threshold, 1.0, 0)
        predictions *= 255

        predictions = predictions.astype(np.uint8)

        return predictions

    def run(self, img_patches, class_threshold: float = 0.6) -> np.array:
        return self._predict(img_patches, class_threshold)

    def run_debug(self, img, class_threshold: float = 0.6):
        predictions = self._predict(img, class_threshold)
        preview_img = self.generate_page_data(img, predictions)

        return predictions, preview_img