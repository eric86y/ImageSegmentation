import os
import numpy as np
import torch
import torch.nn.functional as F

import segmentation_models_pytorch as sm
import pyarrow as pa
import pyarrow.parquet as pq


parquet_schema = pa.schema([
    ("image_name", pa.string()),
    ("image_width", pa.int32()),
    ("image_height", pa.int32()),
    ("num_contours", pa.int32()),

    # contours: list of polygons, polygon = list of (x,y)
    ("contours", pa.list_(
        pa.list_(
            pa.struct([
                ("x", pa.int32()),
                ("y", pa.int32()),
            ])
        )
    )),

    # bounding boxes: one per contour
    ("bboxes", pa.list_(
        pa.struct([
            ("x", pa.int32()),
            ("y", pa.int32()),
            ("w", pa.int32()),
            ("h", pa.int32()),
        ])
    )),
])

def resize_clamp(img: torch.Tensor, patch_size: int = 512, max_w: int =4096, max_h: int = 2048):
    _, H, W = img.shape

    scale_x = 1.0
    scale_y = 1.0

    if W > H and W > max_w:
        scale = max_w / W
    elif H > W and H > max_h:
        scale = max_h / H
    elif H < patch_size:
        scale = patch_size / H
    else:
        return img, scale_x, scale_y

    new_h = int(round(H * scale))
    new_w = int(round(W * scale))

    scale_x = new_w / W
    scale_y = new_h / H

    img = img.unsqueeze(0).float()
    img = torch.nn.functional.interpolate(
        img,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )
    img = img.squeeze(0)

    return img, scale_x, scale_y


def pad_to_multiple(img: torch.Tensor, patch_size=512, value=255):
    _, H, W = img.shape

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    # pad = (left, right, top, bottom)
    img = F.pad(
        img,
        (0, pad_w, 0, pad_h),
        value=value
    )
    return img, pad_w, pad_h


def tile_image(img: torch.Tensor, patch_size=512):
    C, H, W = img.shape
    y_steps = H // patch_size
    x_steps = W // patch_size

    tiles = (
        img
        .unfold(1, patch_size, patch_size)
        .unfold(2, patch_size, patch_size)
    )

    tiles = tiles.permute(1, 2, 0, 3, 4).contiguous()
    tiles = tiles.view(-1, C, patch_size, patch_size)

    return tiles, x_steps, y_steps


def stitch_tiles(
    preds: torch.Tensor,
    x_steps: int,
    y_steps: int,
    patch_size: int = 512,
):
    """
    preds: [N, C, H, W]
    returns: [C, H_full, W_full]
    """
    N, C, H, W = preds.shape
    assert H == patch_size and W == patch_size
    assert N == x_steps * y_steps

    # [N, C, H, W] → [y, x, C, H, W]
    tiles = preds.view(y_steps, x_steps, C, H, W)

    # stitch width
    rows = []
    for y in range(y_steps):
        rows.append(torch.cat(list(tiles[y]), dim=-1))  # concat W

    # stitch height
    full = torch.cat(rows, dim=-2)  # concat H

    return full


def crop_padding(mask: torch.Tensor, pad_x: int, pad_y: int):
    """
    mask: [C, H, W]
    """
    if pad_y > 0:
        mask = mask[:, :-pad_y, :]
    if pad_x > 0:
        mask = mask[:, :, :-pad_x]
    return mask


def normalize(img):
    return img.float().div_(255.0)


def torch_adaptive_threshold(img: torch.Tensor, ksize: int = 51, c: int =13):
    gray = img.mean(dim=1, keepdim=True)
    mean = F.avg_pool2d(gray, ksize, stride=1, padding=ksize//2)
    return (gray > (mean - c)).float() * 255


def collate_fn(batch):
    img, meta = batch[0] # for batch_size == 1

    img, sx, sy = resize_clamp(img)
    img, pad_x, pad_y = pad_to_multiple(img)

    tiles, x_steps, y_steps = tile_image(img)
    tiles = tiles.float().div_(255.0)

    meta["scale_x"] = sx
    meta["scale_y"] = sy
    meta["pad_x"] = pad_x
    meta["pad_y"] = pad_y
    meta["x_steps"] = x_steps
    meta["y_steps"] = y_steps

    return img, tiles, meta


def multi_image_collate_fn(batch):
    all_tiles = []
    tile_ranges = []
    metas = []

    offset = 0

    for img, meta in batch:
        img, sx, sy = resize_clamp(img)
        img, pad_x, pad_y = pad_to_multiple(img)

        tiles, x_steps, y_steps = tile_image(img)
        tiles = tiles.float().div_(255.0)

        n_tiles = tiles.shape[0]
        tile_ranges.append((offset, offset + n_tiles))
        all_tiles.append(tiles)

        meta["scale_x"] = sx
        meta["scale_y"] = sy
        meta["pad_x"] = pad_x
        meta["pad_y"] = pad_y
        meta["x_steps"] = x_steps
        meta["y_steps"] = y_steps

        metas.append(meta)
        offset += n_tiles

    all_tiles = torch.cat(all_tiles, dim=0)

    return all_tiles, tile_ranges, metas


def load_model(checkpoint_path: str, classes: int, device: str):
    checkpoint = torch.load(checkpoint_path)
    
    model = sm.DeepLabV3Plus(classes=classes).to(device)
    model.load_state_dict(checkpoint['state_dict'])

    """model = models.segmentation.deeplabv3_resnet50(
        weights=None,
        num_classes=2
    )"""
    model.to(device)
    model.eval()
    return model


def contour_to_cv(contour):
    """
    contour: list[(x, y)]
    returns: np.ndarray [N, 1, 2] int32
    """
    return np.array(contour, dtype=np.int32).reshape(-1, 1, 2)


def contour_to_original(contour, scale_x: float, scale_y: float):
    return [
        (
            int(round(x / scale_x)),
            int(round(y / scale_y)),
        )
        for x, y in contour
    ]


def bbox_to_original(bbox, scale_x: float, scale_y: float):
    x, y, w, h = bbox
    return (
        int(round(x / scale_x)),
        int(round(y / scale_y)),
        int(round(w / scale_x)),
        int(round(h / scale_y)),
    )


def bboxes_to_pyarrow(bboxes):
    return [
        {"x": x, "y": y, "w": w, "h": h}
        for (x, y, w, h) in bboxes
    ]

def contours_to_arrow(contours):
    return [
        [{"x": x, "y": y} for x, y in contour]
        for contour in contours
    ]


def write_result_parquet(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    base_name, _ = os.path.splitext(result["image_name"])

    table = pa.Table.from_pylist(
        [{
            "image_name": result["image_name"],
            "image_width": result["image_width"],
            "image_height": result["image_height"],
            "num_contours": result["num_contours"],
            "contours": contours_to_arrow(result["contours"]),
            "bboxes": bboxes_to_pyarrow(result["bboxes"]),
        }],
        schema=parquet_schema,
    )

    out_path = os.path.join(out_dir, f"{base_name}.parquet")

    pq.write_table(
        table,
        out_path,
        compression="zstd"
    )
