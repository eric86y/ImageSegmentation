import os
import argparse
from glob import glob
from natsort import natsorted
from pathlib import Path

from Source.Utils import create_dir, sanity_check, split_dataset, generate_tiled_dataset


def collect_dataset(data_root: Path, ok_filter: str = "OK"):
    all_images = []
    all_xml = []

    for sub_dir in data_root.iterdir():
        if ok_filter in sub_dir.name:
            images = natsorted(glob(str(sub_dir / "*.jpg")))
            xml = natsorted(glob(str(sub_dir / "page" / "*.xml")))

            assert len(images) == len(xml), f"Mismatch in {sub_dir}"

            all_images.extend(images)
            all_xml.extend(xml)

    return all_images, all_xml


def main():
    parser = argparse.ArgumentParser(
        description="Build tiled train/val/test dataset from PageXML + images"
    )

    parser.add_argument(
        "-i", "--data-root",
        type=Path,
        required=True,
        help="Root directory containing subfolders with images + page/*.xml",
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <data-root>/MultiClassDataset/Tiled)",
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.8,
        help="Tile overlap ratio (default: 0.8)",
    )

    parser.add_argument(
        "--ok-filter",
        type=str,
        default="OK",
        help="Only include subfolders containing this string (default: 'OK')",
    )

    args = parser.parse_args()

    data_root = args.data_root
    overlap = args.overlap
    ok_filter = args.ok_filter

    if args.output_dir is None:
        output_dir = data_root / "MultiClassDataset" / "Tiled"
    else:
        output_dir = args.output_dir

    # ------------------------------------------------------------
    # Collect dataset
    # ------------------------------------------------------------
    all_images, all_xml = collect_dataset(data_root, ok_filter)

    print(f"Dataset => Images: {len(all_images)}, XML: {len(all_xml)}")

    sanity_check(all_images, all_xml)

    # ------------------------------------------------------------
    # Split
    # ------------------------------------------------------------
    (
        train_images,
        train_xml,
        val_images,
        val_xml,
        test_images,
        test_xml,
    ) = split_dataset(all_images, all_xml)

    print(f"Train => Images: {len(train_images)}, XML: {len(train_xml)}")
    print(f"Val   => Images: {len(val_images)}, XML: {len(val_xml)}")
    print(f"Test  => Images: {len(test_images)}, XML: {len(test_xml)}")

    # ------------------------------------------------------------
    # Output dirs
    # ------------------------------------------------------------
    train_imgs_dir = output_dir / "train" / "images"
    train_masks_dir = output_dir / "train" / "masks"

    val_imgs_dir = output_dir / "val" / "images"
    val_masks_dir = output_dir / "val" / "masks"

    test_imgs_dir = output_dir / "test" / "images"
    test_masks_dir = output_dir / "test" / "masks"

    for d in [
        train_imgs_dir,
        train_masks_dir,
        val_imgs_dir,
        val_masks_dir,
        test_imgs_dir,
        test_masks_dir,
    ]:
        create_dir(d)

    # ------------------------------------------------------------
    # Generate tiled datasets
    # ------------------------------------------------------------
    print("Generating train tiles...")
    generate_tiled_dataset(
        train_images, train_xml,
        train_imgs_dir, train_masks_dir,
        overlap=overlap
    )

    print("Generating val tiles...")
    generate_tiled_dataset(
        val_images, val_xml,
        val_imgs_dir, val_masks_dir,
        overlap=overlap
    )

    print("Generating test tiles...")
    generate_tiled_dataset(
        test_images, test_xml,
        test_imgs_dir, test_masks_dir,
        overlap=overlap
    )

    print("Done.")
    print(f"Output written to: {output_dir}")


if __name__ == "__main__":
    main()