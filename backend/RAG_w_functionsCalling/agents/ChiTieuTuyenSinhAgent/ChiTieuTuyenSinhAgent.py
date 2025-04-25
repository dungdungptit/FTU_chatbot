import os

from RAG_w_functionsCalling.agents.rag import RagMini
from RAG_w_functionsCalling.agents.retriever import Retriever
from RAG_w_functionsCalling.core.models import get_chat_completion

data_path = os.path.join(
    os.getcwd(),
    "data_137",
    "chi_tieu_tuyen_sinh_2025"
)

class ChiTieuTuyenSinhAgent(RagMini):
    def __init__(self):
        self.retriever = Retriever(data_path)
        self.description = """Đây là agent chuyên dụng cho việc hỏi đáp các câu hỏi liên quan đến chỉ tiêu tuyển sinh. Nếu người dùng hỏi về các chỉ tiêu tuyển sinh các năm, thông tin chỉ tiêu tuyển sinh chung thì agent này nên được ưu tiên chọn.
        """
        
        self.output = """
        + Các ngưỡng điểm đảm bảo chất lượng của các nhóm đối tượng xét tuyển có thể điều chỉnh cho phù hợp với quy định và điều kiện thực tế. Nội dung chi tiết quy định của các phương thức xét tuyển sẽ được cụ thể hoá trong Đề án Tuyển sinh năm 2025, được công bố sau khi Quy chế Tuyển sinh Đại học năm 2025 của Bộ Giáo dục và Đào tạo ban hành.
        + Trong trường hợp Quy chế Tuyển sinh Đại học năm 2025 có những điều chỉnh dẫn đến sự ảnh hưởng tới các phương thức tuyển sinh của nhà trường, trường Đại học Ngoại thương sẽ có điều chỉnh kịp thời để phù hợp với Quy chế và đảm bảo quyền lợi thí sinh.           
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


chi_tieu_tuyen_sinh_agent = ChiTieuTuyenSinhAgent()
