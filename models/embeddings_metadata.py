from pydantic import BaseModel
from typing import List, Optional


class EmbeddingMetadata(BaseModel):
    embedding_id: str  # md5(file_hash + columns + sheet + model)
    file_hash: str  # md5 hash of the loaded file (raw)
    file_name: str
    model_key: str
    columns: List[str]
    sheet_name: Optional[str] = None
