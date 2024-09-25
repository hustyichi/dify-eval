import pandas as pd

from dify_eval.dataset.extractor.extractor_base import BaseExtractor
from dify_eval.dataset.model import DatasetItem


class CsvExtractor(BaseExtractor):
    def __init__(
        self,
        file_path: str,
        input_column: str = "question",
        output_column: str = "answer",
        metadata_column: str = "metadata",
        encoding: str = "utf-8",
    ) -> None:
        super().__init__(file_path, input_column, output_column, metadata_column)
        self.df = pd.read_csv(file_path, encoding=encoding)

        if input_column not in self.df.columns:
            raise ValueError("Input column not found")

    def extract(self) -> list[DatasetItem]:
        dataset_items = []
        for idx, row in self.df.iterrows():
            dataset_items.append(
                DatasetItem(
                    input=row.get(self.input_column),
                    expected_output=row.get(self.output_column),
                    metadata=row.get(self.metadata_column),
                )
            )

        return dataset_items
