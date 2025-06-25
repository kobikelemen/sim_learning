from pydantic import BaseModel
from typing import List, Optional, Tuple

class Grasp(BaseModel):
    rotation: List[List[float]]
    translation: List[float]