from abc import ABC, abstractmethod

from dify_eval.dataset.model import DatasetItem


class BaseExtractor(ABC):

    def __init__(
        self,
        file_path: str,
        input_column: str = "question",
        output_column: str = "answer",
        metadata_column: str = "metadata",
        encoding: str = "utf-8",
    ):
        self.file_path = file_path
        self.input_column = input_column
        self.output_column = output_column
        self.metadata_column = metadata_column
        self.encoding = encoding

    @abstractmethod
    def extract(self) -> list[DatasetItem]:
        raise NotImplementedError
