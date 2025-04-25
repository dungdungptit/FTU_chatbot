import os

from RAG_w_functionsCalling.agents.rag import RagMini
from RAG_w_functionsCalling.agents.retriever import Retriever
from RAG_w_functionsCalling.core.models import get_chat_completion

data_path = os.path.join(
    os.getcwd(), 
    "data_137",
    "thong_tin_truong"
)


class WelcomeAgent(RagMini):
    def __init__(self):
        self.description = "Đây là agent chuyên dụng đối với nhiệm vụ chào hỏi, tiếp đón, giới thiệu tính năng hệ thống về trường Đại học Ngoại Thương (FTU). Ngoài ra, agent này còn được sử dụng chuyên dụng cho nhu cầu trả lời câu hỏi về thông tin trường như địa chỉ, thông tin liên hệ, số học sinh,... .Nếu câu hỏi, yêu cầu của người dùng mang tính chào hỏi hoặc yêu cầu cung cấp các thông tin chung về trường đại học Ngoại Thương thì đây là agent nên được ưu tiên. Ví dụ: nếu người dùng hỏi 'Xin chào', 'Tôi có thể hỏi những gì' hay 'giới thiệu về trường', 'Thông tin liên hệ,...' thì hãy chọn agent này."

        self.retriever = Retriever(data_path)

    def get_k_relevant(self, query: str, k: int) -> str:
        docs = self.retriever.retriever.invoke(
            query, config={"k": k}
        )

        return "\n".join([doc.page_content for doc in docs])
    
    def invoke(self, question: str) -> str:
        context = self.get_k_relevant(question, 10)
        response = get_chat_completion(
            task = "welcome", 
            params = {"question": question, "context": context, "output": self.output}
        )

        return response

welcome_agent = WelcomeAgent()


