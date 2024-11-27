import os
import cv2
import math
import random
import logging
import statistics
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from numpy.typing import NDArray
from datetime import datetime
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class Bbox:
    x: int
    y: int
    w: int
    h: int

@dataclass
class LineData:
    contour: List
    bbox: Bbox
    center: Tuple[int, int]

@dataclass
class PerigPrediction:
    images: NDArray
    lines: NDArray
    captions: NDArray
    margins: NDArray

@dataclass
class LayoutData:
    images: List[Bbox]
    text_bboxes: List[Bbox]
    lines: List[LineData]
    captions: List[Bbox]
    margins: List[Bbox]
    predictions: Dict


def get_utc_time() -> str:
    t = datetime.now()
    s = t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    s = s.split(" ")

    return f"{s[0]}T{s[1]}"


def show_sample_pair(img_patch: NDArray, mask_patch: NDArray):
    fig = plt.figure(figsize=(16, 16))
    rows = 1
    columns = 2

    fig.add_subplot(rows, columns, 1)
    plt.imshow(img_patch)
    plt.axis("off")
    plt.title("Image")

    fig.add_subplot(rows, columns, 2)
    plt.imshow(mask_patch)
    plt.axis("off")
    plt.title("Mask")


def show_sample_overlay(img_patch: NDArray, mask_patch: NDArray, alpha: float = 0.4):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Image-Mask overlay")
    plt.imshow(img_patch)
    plt.imshow(mask_patch, alpha=alpha)


def create_dir(dir_name: str) -> None:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_filename(file_path: str) -> str:
    name_segments = os.path.basename(file_path).split(".")[:-1]
    name = "".join(f"{x}." for x in name_segments)
    return name.rstrip(".")


def shuffle(a, b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)

    return list(a), list(b)


def split_dataset(images: List[str], masks: List[str], train_val_split: float = 0.2, val_test_split: float = 0.5, seed: int = 42):
    train_images, valtest_images, train_masks, valtest_masks = train_test_split(images, masks, test_size=train_val_split, random_state=seed)
    val_images, test_images, val_masks, test_masks = train_test_split(valtest_images, valtest_masks, test_size=val_test_split, random_state=seed)

    train_images, train_masks = shuffle(train_images, train_masks)
    val_images, val_masks = shuffle(val_images, val_masks)
    test_images, test_masks = shuffle(test_images, test_masks)

    return train_images, train_masks, val_images, val_masks, test_images, test_masks


def binarize(image: NDArray) -> NDArray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(24, 24))
    image = clahe.apply(image)
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 11
    )
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image

def resize_to_width(image: NDArray, target_width: int) -> NDArray:
    width_ratio = target_width / image.shape[1]
    image = cv2.resize(
        image,
        (target_width, int(image.shape[0] * width_ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    return image


def rotate_from_hough(image: NDArray) -> Tuple[NDArray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=0.2, tileGridSize=(8,8))
    cl_img = clahe.apply(gray)
    blurred = cv2.GaussianBlur(cl_img, (13, 13), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 11)

    lines = cv2.HoughLinesP(
        thresh,
        1,
        np.pi / 180,
        threshold=130,
        minLineLength=40,
        maxLineGap=8
    )

    if lines is None or len(lines) == 0:
        logging.warning(f"No lines found in image, skipping...")

        return image, 0
    
    prev_img = image.copy()
    angles = []
    zero_angles = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.atan2(y2-y1, x2-x1) * 180 / np.pi

        if abs(angle) < 5 and abs(angle) > 0:
            angles.append(angle)

        elif int(angle) == 0:
            zero_angles.append(angle)

        cv2.line(prev_img, (x1, y1), (x2, y2), (100, 100, 0), 3)

    if len(angles) != 0:     
        avg_angle = statistics.median(angles)
        ratio = (len(zero_angles) / len(angles))

        if ratio < 0.5:
            rot_angle = avg_angle
        elif ratio > 0.5 and ratio < 0.9:
            rot_angle = avg_angle / 2
        else:
            rot_angle = 0.0
    else:
        logging.warning("No angle data found in image.")
        rot_angle = 0

    rows, cols = image.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), (rot_angle), 1)
    rotated_img = cv2.warpAffine(image, rot_matrix, (cols, rows), borderValue=(0, 0, 0))

    return rotated_img, rot_angle



def patch_image(
    img: NDArray, patch_size: int = 64, overlap: int = 2, is_mask=False
) -> tuple[list, int]:
    """
    A simple slicing function.
    Expects input_image.shape[0] and image.shape[1] % patch_size = 0
    """

    y_steps = img.shape[0] // patch_size
    x_steps = img.shape[1] // patch_size

    patches = []

    for y_step in range(0, y_steps):
        for x_step in range(0, x_steps):
            x_start = x_step * patch_size
            x_end = (x_step * patch_size) + patch_size

            crop_patch = img[
                y_step * patch_size : (y_step * patch_size) + patch_size, x_start:x_end
            ]
            patches.append(crop_patch)

    return patches, y_steps


def unpatch_image(image, pred_patches: list) -> NDArray:
    patch_size = pred_patches[0].shape[1]

    x_step = math.ceil(image.shape[1] / patch_size)

    list_chunked = [
        pred_patches[i : i + x_step] for i in range(0, len(pred_patches), x_step)
    ]

    final_out = np.zeros(shape=(1, patch_size * x_step))

    for y_idx in range(0, len(list_chunked)):
        x_stack = list_chunked[y_idx][0]

        for x_idx in range(1, len(list_chunked[y_idx])):
            patch_stack = np.hstack((x_stack, list_chunked[y_idx][x_idx]))
            x_stack = patch_stack

        final_out = np.vstack((final_out, x_stack))

    final_out = final_out[1:, :]
    final_out *= 255

    return final_out


def pad_image(
    img: NDArray, patch_size: int = 64, is_mask=False, pad_value: int = 255
) -> Tuple[NDArray, Tuple[float, float]]:
    x_pad = (math.ceil(img.shape[1] / patch_size) * patch_size) - img.shape[1]
    y_pad = (math.ceil(img.shape[0] / patch_size) * patch_size) - img.shape[0]

    if is_mask:
        pad_y = np.zeros(shape=(y_pad, img.shape[1], 3), dtype=np.uint8)
        pad_x = np.zeros(shape=(img.shape[0] + y_pad, x_pad, 3), dtype=np.uint8)
    else:
        pad_y = np.ones(shape=(y_pad, img.shape[1], 3), dtype=np.uint8)
        pad_x = np.ones(shape=(img.shape[0] + y_pad, x_pad, 3), dtype=np.uint8)
        pad_y *= pad_value
        pad_x *= pad_value

    img = np.vstack((img, pad_y))
    img = np.hstack((img, pad_x))

    return img, (x_pad, y_pad)


def unpatch_prediction(prediction: NDArray, y_splits: int) -> NDArray:
    prediction *= 255
    prediction_sliced = np.array_split(prediction, y_splits, axis=0)
    prediction_sliced = [np.concatenate(x, axis=1) for x in prediction_sliced]
    prediction_sliced = np.vstack(np.array(prediction_sliced))

    return prediction_sliced


def load_image(img_path: str) -> Tuple[NDArray, NDArray]:
    try:
        img = cv2.imread(img_path, 1)
        return img
    except BaseException as e:
        logging.error(f"Failed to load image: {img_path}, {e}")


def optimize_countour(cnt, e=0.001):
    epsilon = e * cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, epsilon, True)


def prepare_img_patches(image: NDArray) -> NDArray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    image /= 255.0
    image = np.dstack([image, image, image])
    return image
