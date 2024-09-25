from pathlib import Path

from dify_eval.dataset.extractor.csv_extractor import CsvExtractor
from dify_eval.dataset.model import DatasetItem


def extract(
    file_path: str,
    input_column: str = "question",
    output_column: str = "answer",
    metadata_column: str = "metadata",
    encoding: str = "utf-8",
) -> list[DatasetItem]:

    input_file = Path(file_path)
    file_extension = input_file.suffix.lower()

    extractor = None
    if file_extension == ".csv":
        extractor = CsvExtractor(
            file_path, input_column, output_column, metadata_column, encoding
        )
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    return extractor.extract()
