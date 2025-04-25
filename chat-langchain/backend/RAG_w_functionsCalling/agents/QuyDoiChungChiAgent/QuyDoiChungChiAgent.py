import os

from RAG_w_functionsCalling.agents.rag import RagMini
from RAG_w_functionsCalling.agents.retriever import Retriever
from RAG_w_functionsCalling.core.models import get_chat_completion

data_path = os.path.join(os.getcwd(), "data_137", "quy_doi_chung_chi_quoc_te")


class QuyDoiChungChiAgent(RagMini):
    def __init__(self):
        self.retriever = Retriever(data_path)
        self.description = """Đây là agent chuyên dụng cho việc hỏi đáp các câu hỏi liên quan đến quy đổi chứng chỉ quốc tế. Nếu người dùng hỏi về các quy đổi chứng chỉ quốc tế, chứng chỉ đánh giá năng lực và mức quy đổi thì agent này nên được ưu tiên chọn.
        """

        self.output = """
    a. Đối với các phương thức tuyển sinh có sử dụng kết quả học tập THPT và  kết quả thi tốt nghiệp THPT năm 2025
        - Trường hợp tại ngưỡng điểm đánh giá hồ sơ xác định trúng tuyển của từng chương trình đào tạo, số thí sinh nhiều hơn số chỉ tiêu xét tuyển còn lại, Trường sử dụng tiêu chí phụ là điểm thi tốt nghiệp THPT năm 2025 môn Toán. 
        - Riêng đối với phương thức xét tuyển theo chứng chỉ A-Level, tiêu chí phụ là điểm tuyệt đối môn Toán trong kỳ thi xét chứng chỉ A-Level (PUM range).

    b. Đối với phương thức tuyển sinh dựa trên kết quả các kỳ thi đánh giá năng lực  trong nước 
        - Trường hợp tại ngưỡng điểm đánh giá hồ sơ xác định trúng tuyển của từng chương trình đào tạo, số thí sinh nhiều hơn số chỉ tiêu xét tuyển còn lại, Trường sử dụng tiêu chí phụ là kết quả điểm tuyệt đối của bài thi đánh giá năng lực .

    c. Đối với phương thức tuyển sinh dựa trên kết quả các kỳ thi đánh giá năng lực  quốc tế
        - Chứng chỉ A-Level : Trường hợp tại ngưỡng điểm đánh giá hồ sơ xác định trúng tuyển của từng chương trình đào tạo, số thí sinh nhiều hơn số chỉ tiêu xét tuyển còn lại, Trường sử dụng tiêu chí phụ là điểm tuyệt đối môn Toán trong kỳ thi xét chứng chỉ A-Level (PUM range).
        - Chứng chỉ SAT và ACT: Trường hợp tại ngưỡng điểm đánh giá hồ sơ xác định trúng tuyển của từng chương trình đào tạo, số thí sinh nhiều hơn số chỉ tiêu xét tuyển còn lại, Trường sử dụng tiêu chí phụ là điểm tuyệt đối của bài thi SAT và ACT.
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


quy_doi_chung_chi_agent = QuyDoiChungChiAgent()
