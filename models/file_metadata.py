from pydantic import BaseModel
from typing import List, Optional


class FileMetadata(BaseModel):
    file_id: str
    file_name: str
    model_key: str
    columns: List[str]
    sheet_name: Optional[str]  # ← allow None
