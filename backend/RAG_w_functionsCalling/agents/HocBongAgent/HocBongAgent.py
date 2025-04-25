import os

from RAG_w_functionsCalling.agents.rag import RagMini
from RAG_w_functionsCalling.agents.retriever import Retriever
from RAG_w_functionsCalling.core .models import get_chat_completion

data_path = os.path.join(
    os.getcwd(), 
    "data_137", 
    "hoc_bong"
)

class HocBongAgent(RagMini):
    def __init__(self):
        self.retriever = Retriever(data_path)
        self.description = """Đây là agent chuyên dụng cho việc hỏi đáp các câu hỏi liên quan đến học bổng cung cấp bởi trường đại học Ngoại Thương. Nếu người dùng hỏi về các loại học bổng của trường thì agent này nên được ưu tiên chọn
        """
        
    def get_k_relevant(self, query: str, k: int) -> str:
        docs = self.retriever.retriever.invoke(
            query, config={"k": k}
        )

        return "\n".join([doc.page_content for doc in docs])
    

    def invoke(self, question: str) -> str:
        context = self.get_k_relevant(question, 8)
        response = get_chat_completion(
            task = "rag", 
            params = {"question": question, "context": context, "output": self.output}
        )

        return response

hoc_bong_agent = HocBongAgent()

        


