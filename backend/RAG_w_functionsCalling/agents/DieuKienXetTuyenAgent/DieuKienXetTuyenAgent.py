import os

from RAG_w_functionsCalling.agents.rag import RagMini
from RAG_w_functionsCalling.agents.retriever import Retriever
from RAG_w_functionsCalling.core.models import get_chat_completion

data_path = os.path.join(
    os.getcwd(), 
    "data_137",
    "dieu_kien_xet_tuyen"
)


class DieuKienXetTuyenAgent(RagMini):
    def __init__(self):
        self.retriever = Retriever(data_path)
        self.description = """Đây là agent chuyên dụng cho việc hỏi đáp các câu hỏi liên quan đến điều kiện xét tuyển. Nếu người dùng hỏi về các điều kiện xét tuyển của các ngành học, phương thức xét tuyển thì agent này nên được ưu tiên chọn
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
    - Không còn xét tuyển sớm, phải dùng kết quả học tập cả năm lớp 12 để xét tuyển
    - Quy chế mới cũng quy định khi sử dụng kết quả học tập cấp THPT để xét tuyển phải dùng kết quả học tập cả năm lớp 12 của thí sinh. 
    - Ngoài ra, nhằm bảo đảm sự đóng góp của kết quả học tập cả năm lớp 12 không quá thấp trong khi tính điểm xét, Quy chế quy định trọng số tính điểm xét của kết quả học năm lớp 12 không dưới 25%.
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


dieu_kien_xet_tuyen_agent = DieuKienXetTuyenAgent()
