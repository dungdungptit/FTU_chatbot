import os

from RAG_w_functionsCalling.agents.rag import RagMini
from RAG_w_functionsCalling.agents.retriever import Retriever
from RAG_w_functionsCalling.core.models import get_chat_completion

data_path = os.path.join(
    os.getcwd(),  
    "data_137",
    "hoc_phi"
)

class LePhiAgent(RagMini):
    def __init__(self):
        self.retriever = Retriever(data_path)
        self.description = """Đây là agent chuyên dụng cho việc hỏi đáp các câu hỏi liên quan đến lệ phí và học phí của trường. Nếu người dùng hỏi về lệ phí ký túc xá, học phí dự kiến thì agent này nên được ưu tiên chọn.
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
            params = {"question": question, "context": context, "output": "\n".join(self.output)}
        )

        return response

le_phi_agent = LePhiAgent()

        


