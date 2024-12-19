from pydantic import BaseModel
from typing import List

# Input schema
class TimeSeriesInput(BaseModel):
    demanda_lista: List[float] 