from typing import *

from pydantic import BaseModel, Field

class RagResponse(BaseModel):
    response: str = Field(..., description = "Câu trả lời của hệ thống cho câu hỏi của người dùng")
    recommendations: List[Optional[str]] = Field(..., description = "Các câu hỏi gợi ý của hệ thống cho người dùng")