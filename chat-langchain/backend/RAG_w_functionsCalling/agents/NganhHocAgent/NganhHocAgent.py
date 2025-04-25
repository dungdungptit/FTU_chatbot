import os

from RAG_w_functionsCalling.agents.rag import RagMini
from RAG_w_functionsCalling.agents.retriever import Retriever
from RAG_w_functionsCalling.core .models import get_chat_completion

data_path = os.path.join(
    os.getcwd(), 
    "data_137", 
    "nganh_va_chuong_trinh_hoc"
)

class NganhVaChuyenNganhAgent(RagMini):
    def __init__(self):
        self.retriever = Retriever(data_path)
        self.description = """Đây là agent chuyên dụng cho việc hỏi đáp các câu hỏi liên quan đến các ngành học và chuyên ngành của trường đại học Ngoại Thương. Nếu người dùng hỏi các thông tin về ngành học cụ thể, chuyên ngành học thì agent này nên được ưu tiên chọn.
        """

    def get_k_relevant(self, query: str, k: int) -> str:
        docs = self.retriever.retriever.invoke(
            query, config={"k": k}
        )

        return "\n".join([doc.page_content for doc in docs])
    

    def invoke(self, question: str) -> str:
        context = self.get_k_relevant(question, 15)
        response = get_chat_completion( 
            task = "rag", 
            params = {"question": question, "context": context, "output": self.output}
        )

        return response

nganh_va_chuyen_nganh_agent = NganhVaChuyenNganhAgent()

        


