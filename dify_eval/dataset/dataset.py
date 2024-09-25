import os

from dotenv import load_dotenv
from langfuse import Langfuse

from dify_eval.dataset.extractor import extractor
from dify_eval.dataset.model import DatasetItem

load_dotenv()

langfuse = Langfuse()


def create_dataset_items(
    dataset_name: str, dataset_items: list[DatasetItem] | None = None
):
    if not dataset_items:
        return

    for item in dataset_items:
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input=item.input,
            expected_output=item.expected_output,
            metadata=item.metadata,
        )


def create_dataset(
    dataset_name: str = os.getenv("DATASET_NAME", ""),
    dataset_items: list[DatasetItem] | None = None,
    always_add_dataset_items: bool = False,
):
    if not dataset_name:
        raise ValueError("Dataset name is required")

    try:
        langfuse.get_dataset(dataset_name)
        if always_add_dataset_items:
            create_dataset_items(dataset_name, dataset_items)
    except Exception:
        langfuse.create_dataset(name=dataset_name)
        create_dataset_items(dataset_name, dataset_items)


def create_dataset_from_file(
    file_path: str = os.getenv("LOCAL_FILE_PATH", ""),
    input_column: str = "question",
    output_column: str = "answer",
    metadata_column: str = "metadata",
    encoding: str = "utf-8",
    dataset_name: str = os.getenv("DATASET_NAME", ""),
    always_add_dataset_items: bool = False,
):
    if not (file_path and os.path.exists(file_path)):
        return

    dataset_items = extractor.extract(
        file_path, input_column, output_column, metadata_column, encoding
    )

    return create_dataset(dataset_name, dataset_items, always_add_dataset_items)
