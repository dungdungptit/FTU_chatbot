import os

from RAG_w_functionsCalling.agents.rag import RagMini
from RAG_w_functionsCalling.agents.retriever import Retriever
from RAG_w_functionsCalling.core.models import get_chat_completion

data_path = os.path.join(
    os.getcwd(), 
    "data_137",
    "ky_tuc_xa"
)

class KyTucXaAgent(RagMini):
    def __init__(self):
        self.retriever = Retriever(data_path)
        self.description = """Đây là agent chuyên dụng cho việc hỏi đáp các câu hỏi liên quan đến ký túc xá. Nếu người dùng hỏi về các thông tin của ký túc xá như cách đăng ký, hồ sơ đăng ký, hỗ trợ miễn giảm chi phí,... thì agent này nên được ưu tiên chọn
        """

        self.output = """
    + Quy định này áp dụng cho sinh viên chính quy đang theo học tại Trường Đại học Ngoại thương, có nhu cầu ở Ký túc xá  của trường tại 91 phố Chùa Láng, từ năm học 2008 - 2009 (bắt đầu từ khóa 47 hệ đại học và khóa 4 hệ cao đẳng).
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

ky_tuc_xa_agent = KyTucXaAgent()

        


