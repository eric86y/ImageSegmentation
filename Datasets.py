

from zipfile import ZipFile
from huggingface_hub import snapshot_download


def _do_snapshot_download(dataset_id = str, cache_dir: str = "Datasets") -> str | None:
    try:
        data_path = snapshot_download(
                repo_id=dataset_id,
                repo_type="dataset",
                cache_dir=cache_dir)
        return data_path
    
    except BaseException as e:
        print(f"Error: {e}")
        return None

def download_line_dataset_light(target_dir: str = "Datasets") -> str:
    dataset_id = "Eric-23xd/TibetanLineDetection_light"

    data_path = _do_snapshot_download(dataset_id, target_dir)

    with ZipFile(f"{data_path}/data.zip", 'r') as zip:
        zip.extractall(data_path)

    return data_path



def download_line_dataset(target_dir: str = "Datasets") -> str:
    dataset_id = "BDRC/LineSegmentation"
    data_path = _do_snapshot_download(dataset_id, target_dir)

    with ZipFile(f"{data_path}/PhotiLines.zip", 'r') as zip:
        zip.extractall(data_path)

    return data_path


def download_photi_layout_dataset(target_dir: str = "Datasets") -> str:
    dataset_id = "BDRC/LayoutSegmentation_Dataset"
    data_path = _do_snapshot_download(dataset_id, target_dir)

    with ZipFile(f"{data_path}/data.zip", 'r') as zip:
        zip.extractall(data_path)

    return data_path


def download_modern_books_datset(target_dir: str = "Datasets") -> str:
    dataset_id = "BDRC/ModernBooksLayout_v1"
    data_path = _do_snapshot_download(dataset_id, target_dir)

    with ZipFile(f"{data_path}/data.zip", 'r') as zip:
        zip.extractall(data_path)

    return data_path
