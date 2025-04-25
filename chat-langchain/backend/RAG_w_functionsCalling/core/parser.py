from langchain_core.output_parsers import PydanticOutputParser

from RAG_w_functionsCalling.schemas import (
    rag_response,
    function_calling_response,
)

rag_parser = PydanticOutputParser(pydantic_object=rag_response.RagResponse)
function_calling_parser = PydanticOutputParser(pydantic_object=function_calling_response.FunctionCallingResponse)