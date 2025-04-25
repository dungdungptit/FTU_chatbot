import os

from RAG_w_functionsCalling.agents.rag import RagMini
from RAG_w_functionsCalling.agents.retriever import Retriever
from RAG_w_functionsCalling.core.models import get_chat_completion

data_path = os.path.join(
    os.getcwd(), 
    "data_137",
    "tuyen_sinh"
)

class TuyenSinhAgent(RagMini):
    def __init__(self):
        self.retriever = Retriever(data_path)
        self.description = """Đây là agent được sử dụng cho việc hỏi đáp các câu hỏi liên quan đến tuyển sinh, tuy nhiên chỉ chọn agent này nếu không agent nào được ưu tiên phù hợp hơn cho câu hỏi.
        """

    def get_k_relevant(self, query: str, k: int) -> str:
        docs = self.retriever.retriever.invoke(
            query, config={"k": k}
        )

        return "\n".join([doc.page_content for doc in docs])
    

    def invoke(self, question: str) -> str:
        context = self.get_k_relevant(question, 10)
        response = get_chat_completion( 
            task = "rag", 
            params = {"question": question, "context": context, "output": self.output}
        )

        return response

tuyen_sinh_agent = TuyenSinhAgent()

        


