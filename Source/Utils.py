import cv2
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import statistics

from datetime import datetime, timezone
from glob import glob
from natsort import natsorted
from numpy.typing import NDArray
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from xml.dom import minidom

from Config import COLOR_DICT
from Source.Data import BRect



def get_utc_time() -> str:
    t = datetime.now(timezone.utc)
    s = t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    date, time = s.split(" ")
    return f"{date}T{time}Z"



def show_image(
    image: NDArray, cmap: str = "", axis="off", fig_x: int = 24, fix_y: int = 13
) -> None:
    plt.figure(figsize=(fig_x, fix_y))
    plt.axis(axis)

    if cmap != "":
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)


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


def show_sample_overlay(
    img_patch: NDArray,
    mask_patch: NDArray,
    fig_x: int = 24,
    fix_y: int = 13,
    alpha: float = 0.4,
):
    plt.figure(figsize=(fig_x, fix_y))
    plt.axis("off")
    plt.title("Image-Mask overlay")
    plt.imshow(img_patch)
    plt.imshow(mask_patch, alpha=alpha)


def create_dir(dir_path: str) -> None:
    try:
        os.makedirs(dir_path, exist_ok=True)
    except BaseException as e:
        logging.error(f"Failed creating output directories: {e}")


def get_filename(file_path: str) -> str:
    name_segments = os.path.basename(file_path).split(".")[:-1]
    name = "".join(f"{x}." for x in name_segments)
    return name.rstrip(".")


def shuffle(a, b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)

    return list(a), list(b)


def split_dataset_simple(
    images: list[str],
    annotations: list[str],
    split_ratio: float = 0.2,
    seed: int = 42,
):
    train_images, val_images, train_xml, val_xml = train_test_split(
        images, annotations, test_size=split_ratio, random_state=seed
    )

    return train_images, train_xml, val_images, val_xml


def split_dataset(
    images: list[str],
    masks: list[str],
    train_val_split: float = 0.2,
    val_test_split: float = 0.5,
    seed: int = 42,
):
    train_images, valtest_images, train_masks, valtest_masks = train_test_split(
        images, masks, test_size=train_val_split, random_state=seed
    )
    val_images, test_images, val_masks, test_masks = train_test_split(
        valtest_images, valtest_masks, test_size=val_test_split, random_state=seed
    )

    train_images, train_masks = shuffle(train_images, train_masks)
    val_images, val_masks = shuffle(val_images, val_masks)
    test_images, test_masks = shuffle(test_images, test_masks)

    return train_images, train_masks, val_images, val_masks, test_images, test_masks


def crop_image_to_bbox(image: NDArray, mask: NDArray) -> tuple[NDArray, NDArray]:
    brect = get_brect(mask)
    image = image[brect.y : brect.y + brect.h, brect.x : brect.x + brect.w]
    mask = mask[brect.y : brect.y + brect.h, brect.x : brect.x + brect.w]

    return image, mask


def binarize(image: NDArray) -> NDArray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(24, 24))
    image = clahe.apply(image)
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 11
    )
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image


def rotate_from_hough(image: NDArray) -> tuple[NDArray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=0.2, tileGridSize=(8, 8))
    cl_img = clahe.apply(gray)
    blurred = cv2.GaussianBlur(cl_img, (13, 13), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 11
    )

    lines = cv2.HoughLinesP(
        thresh, 1, np.pi / 180, threshold=130, minLineLength=40, maxLineGap=8
    )

    if lines is None or len(lines) == 0:
        logging.warning("No lines found in image, skipping...")

        return image, 0

    prev_img = image.copy()
    angles = []
    zero_angles = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi

        if abs(angle) < 5 and abs(angle) > 0:
            angles.append(angle)

        elif int(angle) == 0:
            zero_angles.append(angle)

        cv2.line(prev_img, (x1, y1), (x2, y2), (100, 100, 0), 3)

    if len(angles) != 0:
        avg_angle = statistics.median(angles)
        ratio = len(zero_angles) / len(angles)

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
    img: NDArray, patch_size: int = 64, overlap: int = 2, is_mask: bool = False
) -> tuple[list[NDArray], int]:
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


def pad_to_multiple(img: NDArray, multiple: int = 512):
    """adaptively padding the images to the chosen tile size"""
    h, w = img.shape[:2]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    pad_h = new_h - h
    pad_w = new_w - w
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)


def pad_image(
    img: NDArray, patch_size: int = 64, is_mask: bool = False, pad_value: int = 255
) -> tuple[NDArray, tuple[float, float]]:
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


def load_image(img_path: str) -> tuple[NDArray, NDArray]:
    try:
        img = cv2.imread(img_path, 1)
        return img
    except BaseException as e:
        logging.error(f"Failed to load image: {img_path}, {e}")


def is_mask_empty(mask: NDArray):
    """Prüft, ob Maske komplett schwarz ist."""
    if len(mask.shape) == 3:
        return not np.any(mask)
    else:

        return not np.any(mask > 0)


def optimize_countour(cnt, e: float = 0.001):
    epsilon = e * cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, epsilon, True)


def prepare_img_patches(image: NDArray) -> NDArray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    image /= 255.0
    image = np.dstack([image, image, image])

    return image


"""
Segmentation Mask Generation

"""


def get_brect(image: NDArray) -> BRect:
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.boundingRect(image)

    return BRect(x, y, w, h)


def get_color(key: str) -> list[int]:
    color = COLOR_DICT[key]
    color = color.split(",")
    color = [x.strip() for x in color]
    color = [int(x) for x in color]

    return color


def sanity_check(images: list[str], annotations: list[str]) -> None:
    image_names = [os.path.basename(x).split(".")[0] for x in images]
    annotation_names = [os.path.basename(x).split(".")[0] for x in annotations]

    for image_n, annotation_n in zip(image_names, annotation_names):
        assert image_n == annotation_n


def check_xml_status(xml_file: str) -> bool:
    annotation_tree = minidom.parse(xml_file)
    doc_metadata = annotation_tree.getElementsByTagName("TranskribusMetadata")

    if len(doc_metadata) > 0:
        page_status = doc_metadata[0].attributes["status"].value

        if page_status == "DONE":
            return True

        else:
            return False

    else:
        return False


def get_xml_point_list(attribute: str) -> NDArray:
    """
    parses the PageXML Coords element and returns an np.array that conforms cv2 contours
    """
    coords = attribute.getElementsByTagName("Coords")
    base_points = coords[0].attributes["points"].value
    pts = base_points.split(" ")
    pts = [x for x in pts if x != ""]

    points = []
    for p in pts:
        x, y = p.split(",")
        a = int(float(x)), int(float(y))
        points.append(a)

    point_array = np.array(points, dtype=np.int32)

    return point_array


def resize_to_width(image: NDArray, target_width: int) -> tuple[NDArray, float]:
    ratio = target_width / image.shape[1]
    image = cv2.resize(
        image,
        (target_width, int(image.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    return image, ratio


"""
Segmentation Mask Generation
"""


def generate_dataset(
    images: list[str],
    annotations: list[str],
    img_out_dir: str,
    mask_out_dir: str,
    precrop: bool = False
):
    assert len(images) == len(annotations)

    for _img, _xml in tqdm(zip(images, annotations), total=len(images)):
        image_n = os.path.basename(_img).split(".")[0]
        img, mask = generate_mask_image(_img, _xml)

        if precrop:
            crop_image_to_bbox(img, mask)

        if img is None or mask is None:
            print(f"Warning: Error processing {_img} or {_xml}.")
            continue

        if img.shape[:2] != mask.shape[:2]:
            print(f"Warning: image and mask have different sizes {_img}")
            continue

        img_out = f"{img_out_dir}/{image_n}.jpg"
        mask_out = f"{mask_out_dir}/{image_n}_mask.png"

        shutil.copy2(_img, img_out)
        cv2.imwrite(mask_out, mask)


def generate_tiled_dataset(
    images: list[str],
    annotations: list[str],
    img_out_dir: str,
    mask_out_dir: str,
    tile_size: int = 512,
    overlap: float = 0.9,
    precrop: bool = False,
):
    stride = int(tile_size * (1 - overlap))

    assert stride != 0

    for _img, _xml in tqdm(zip(images, annotations), total=len(images)):
        img, mask = generate_mask_image(_img, _xml)

        if precrop:
            crop_image_to_bbox(img, mask)

        img = pad_to_multiple(img, tile_size)
        mask = pad_to_multiple(mask, tile_size)

        h, w = img.shape[:2]

        tile_idx = 0
        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                img_tile = img[y : y + tile_size, x : x + tile_size]
                mask_tile = mask[y : y + tile_size, x : x + tile_size]

                # skip if mask is empty (i.e. black area)
                if is_mask_empty(mask_tile):
                    continue

                base_name = Path(_img).stem
                tile_name = f"{base_name}_y{y}_x{x}.png"

                cv2.imwrite(os.path.join(img_out_dir, tile_name), img_tile)
                cv2.imwrite(os.path.join(mask_out_dir, tile_name), mask_tile)

                tile_idx += 1


def generate_multi_mask(
    img: NDArray, annotation_file: str, annotate_lines: bool
) -> NDArray:
    try:
        annotation_tree = minidom.parse(annotation_file)
    except BaseException as e:
        print(f"Failed to parse: {annotation_file}, {e}")
        return

    img_height = img.shape[0]
    img_width = img.shape[1]
    image_mask = np.ones(shape=(int(img_height), int(img_width), 3), dtype=np.uint8)

    textareas = annotation_tree.getElementsByTagName("TextRegion")
    imageareas = annotation_tree.getElementsByTagName("ImageRegion")
    line_areas = annotation_tree.getElementsByTagName("TextLine")
    sep_areas = annotation_tree.getElementsByTagName(
        "UnknownRegion"
    )  # used for black bars between header and text lines

    cv2.floodFill(
        image=image_mask, mask=None, seedPoint=(0, 0), newVal=get_color("background")
    )

    if len(textareas) != 0:
        for text_area in textareas:
            area_attrs = text_area.attributes["custom"].value

            if "marginalia" in area_attrs:
                cv2.fillPoly(
                    image_mask,
                    [get_xml_point_list(text_area)],
                    color=get_color("margin"),
                )

            # handles cases in which the annotators labelled images via a Textarea with "image" tag
            elif "image" in area_attrs:
                cv2.fillPoly(
                    image_mask,
                    [get_xml_point_list(text_area)],
                    color=get_color("image"),
                )

            elif "caption" in area_attrs:
                cv2.fillPoly(
                    image_mask,
                    [get_xml_point_list(text_area)],
                    color=get_color("caption"),
                )
            elif "page-number" in area_attrs:
                cv2.fillPoly(
                    image_mask,
                    [get_xml_point_list(text_area)],
                    color=get_color("pagenr"),
                )
            elif "footer" in area_attrs:
                cv2.fillPoly(
                    image_mask,
                    [get_xml_point_list(text_area)],
                    color=get_color("footer"),
                )
            elif "header" in area_attrs:
                cv2.fillPoly(
                    image_mask,
                    [get_xml_point_list(text_area)],
                    color=get_color("header"),
                )
            elif "heading" in area_attrs:
                cv2.fillPoly(
                    image_mask,
                    [get_xml_point_list(text_area)],
                    color=get_color("header"),
                )
            elif "table" in area_attrs:
                cv2.fillPoly(
                    image_mask,
                    [get_xml_point_list(text_area)],
                    color=get_color("table"),
                )

            else:
                if not annotate_lines:
                    cv2.fillPoly(
                        image_mask,
                        [get_xml_point_list(text_area)],
                        color=get_color("text"),
                    )

    if len(imageareas) != 0:
        for img in imageareas:
            cv2.fillPoly(
                image_mask, [get_xml_point_list(img)], color=get_color("image")
            )

    if len(sep_areas) != 0:
        for sep in sep_areas:
            cv2.fillPoly(
                image_mask, [get_xml_point_list(sep)], color=get_color("separator")
            )

    if annotate_lines:
        if len(line_areas) != 0:
            for line in line_areas:
                cv2.fillPoly(
                    image_mask, [get_xml_point_list(line)], color=get_color("line")
                )

    return image_mask


def generate_mask_image(
    image_path: str, xml_path: str, annotate_lines: bool = True
) -> tuple[NDArray, NDArray]:
    img = cv2.imread(image_path)
    clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(24, 24))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = clahe.apply(img)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 11
    )

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    mask = generate_multi_mask(img, xml_path, annotate_lines)

    return img, mask


def generate_masks(
    directory: str, annotate_lines: str = "yes", overlay_preview: str = "no"
) -> None:
    """
    args:
    - overlay_preview: creates an overlay of the original image and the mask for debugging purposes
    - annotate lines: uses the "TextLine" Element in order to draw the line boxes instead of the TextArea
    """

    _images = natsorted(glob(f"{directory}/*.jpg"))
    _xml = natsorted(glob(f"{directory}/page/*.xml"))

    try:
        sanity_check(_images, _xml)
    except BaseException as e:
        logging.error(f"Image-Label Pairing broken in: {directory}, {e}")
        return

    mask_dir = os.path.join(directory, "Masks")
    output_dir = os.path.join(mask_dir, "Multiclass")
    output_masks = os.path.join(output_dir, "Masks")

    create_dir(mask_dir)
    create_dir(output_dir)
    create_dir(output_masks)

    logging.info(f"created output directory: {output_dir}")

    for _img, _xml in tqdm(zip(_images, _xml), total=len(_images)):
        image_n = os.path.basename(_img).split(".")[0]
        img, mask = generate_mask_image(_img, _xml, annotate_lines)

        mask_out = f"{output_masks}/{image_n}_mask.png"

        if overlay_preview == "yes":
            cv2.addWeighted(mask, 0.4, img, 1 - 0.4, 0, img)
            cv2.imwrite(mask_out, img)
        else:
            cv2.imwrite(mask_out, mask)
