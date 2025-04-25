from typing import *

from pydantic import BaseModel, Field

class FunctionCallingResponse(BaseModel):
    response: str = Field(..., description = "Tên function được chọn")
    question: str = Field(..., description = "Câu hỏi của người dùng")

class FunctionsCallingResponse(BaseModel):
    response: List[FunctionCallingResponse] = Field(..., description = "Danh sách các function được chọn")