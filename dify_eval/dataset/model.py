from typing import Any

import pydantic


class DatasetItem(pydantic.BaseModel):
    input: str | dict
    expected_output: str | None
    metadata: Any
