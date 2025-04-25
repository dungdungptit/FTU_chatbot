import os

from RAG_w_functionsCalling.agents.rag import RagMini
from RAG_w_functionsCalling.agents.retriever import Retriever
from RAG_w_functionsCalling.core.models import get_chat_completion

data_path = os.path.join(os.getcwd(), "data_137", "thay_doi_moi_cua_2025")


class ChinhSachMoiAgent(RagMini):
    def __init__(self):
        self.retriever = Retriever(data_path)
        self.description = """Đây là agent chuyên dụng cho trả lời các câu hỏi liên quan đến chính sách mới của trường đại học Ngoại Thương. Nếu người dùng hỏi về các chính sách mới của trường, agent này nên được ưu tiên.
        """

        self.output = """
- Với mỗi chính sách được cung cấp trong phần Thông tin cung cấp, bạn hãy nêu rõ các trường thông tin:
    + Tên/ tiêu đề của chính sách
    + Thời gian ban hành
    + Lĩnh vực áp dụng
    + Đối tượng
    + Nêu rõ thông tin hiện có
"""

    def get_k_relevant(self, query: str, k: int) -> str:
        docs = self.retriever.retriever.invoke(query, config={"k": k})

        return "\n".join([doc.page_content for doc in docs])

    def invoke(self, question: str) -> str:
        context = self.get_k_relevant(question, 8)
        response = get_chat_completion(
            task="rag",
            params={"question": question, "context": context, "output": self.output},
        )

        return response


chinh_sach_moi_agent = ChinhSachMoiAgent()
