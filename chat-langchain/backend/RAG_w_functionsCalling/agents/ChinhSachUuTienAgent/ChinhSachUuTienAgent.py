import os

from RAG_w_functionsCalling.agents.rag import RagMini
from RAG_w_functionsCalling.agents.retriever import Retriever
from RAG_w_functionsCalling.core.models import get_chat_completion

data_path = os.path.join(os.getcwd(), "data_137", "chinh_sach_uu_tien")


class ChinhSachUuTienAgent(RagMini):
    def __init__(self):
        self.retriever = Retriever(data_path)
        self.description = """Đây là agent chuyên dụng cho việc trả lời các câu hỏi liên quan đến các chính ưu tiên của trường đại học Ngoại Thương. Nếu người dùng hỏi các câu hỏi liên quan đến chính sách ưu tiên của đối tượng, khu vực hay chính sách ưu tiên cho ký túc xá,... agent này nên được ưu tiên."""

        self.output = """
    + Khu vực tuyển sinh của mỗi thí sinh được xác định theo địa điểm trường mà thí sinh đã học lâu nhất trong thời gian học cấp THPT (hoặc trung cấp); nếu thời gian học (dài nhất) tại các khu vực tương đương nhau thì xác định theo khu vực của trường mà thí sinh theo học sau cùng;
    + Từ năm 2023, thí sinh được hưởng chính sách ưu tiên khu vực theo quy định trong năm tốt nghiệp THPT (hoặc trung cấp) và một năm kế tiếp.
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


chinh_sach_uu_tien_agent = ChinhSachUuTienAgent()
