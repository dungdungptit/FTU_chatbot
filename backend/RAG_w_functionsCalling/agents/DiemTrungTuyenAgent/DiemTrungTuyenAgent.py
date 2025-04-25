import os

from RAG_w_functionsCalling.agents.rag import RagMini
from RAG_w_functionsCalling.agents.retriever import Retriever
from RAG_w_functionsCalling.core.models import get_chat_completion

data_path = os.path.join(
    os.getcwd(), 
    "data_137",
    "diem_trung_tuyen"
)


class DiemTrungTuyenAgent(RagMini):
    def __init__(self):
        self.retriever = Retriever(data_path)
        self.description = """Đây là agent chuyên dụng cho việc hỏi đáp các câu hỏi liên quan đến điểm trúng tuyển của các ngành học. Nếu người dùng hỏi về các điểm trúng tuyển của ngành học, phương thức xét tuyển của các năm thì agent này nên được ưu tiên chọn"""

        self.output = """
    + Thí sinh không cần chọn mã phương thức, mã tổ hợp… chỉ cần xác định rõ chương trình, ngành, nhóm ngành đào tạo và cơ sở đào tạo mong muốn theo học để quyết định đăng ký. Hệ thống hỗ trợ tuyển sinh chung của Bộ GDĐT sẽ sử dụng phương thức có kết quả cao nhất của thí sinh để xét tuyển.
    """

    def get_k_relevant(self, query: str, k: int) -> str:
        docs = self.retriever.retriever.invoke(query, config={"k": k})

        return "\n".join([doc.page_content for doc in docs])

    def invoke(self, question: str) -> str:
        context = self.get_k_relevant(question, 15)
        response = get_chat_completion(
            task="rag",
            params={"question": question, "context": context, "output": self.output},
        )

        return response


diem_trung_tuyen_agent = DiemTrungTuyenAgent()
