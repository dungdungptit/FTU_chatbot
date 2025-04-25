from dotenv import load_dotenv
from urllib.parse import urlparse
from langchain_community.utilities import GoogleSerperAPIWrapper

from RAG_w_functionsCalling.core.models import get_chat_completion

load_dotenv()

class SearchAgent:
    def __init__(self):
        self.description = """
            Đây là agent chuyên dụng cho mục đích search web để trả lời cho câu hỏi người dùng. 
            Nếu người dùng đề cập đến việc tra cứu trên mạng để tìm kiếm thông tin, agent này nên được ưu tiên.
        """
        self.search_tool = GoogleSerperAPIWrapper()
    
    def invoke(self, question):
        """Thực hiện tìm kiếm và trả về kết quả từ ftu.edu.vn"""
        snippets, link_title = self._search_ftu(question)
        if not snippets:
            snippets = "Không tìm thấy kết quả nào từ ftu.edu.vn."
        
        summary = get_chat_completion(
            task = "search",
            params = {"question": question, "snippets": snippets}
        )
        
        return {
            "response": summary["response"],
            "recommendations": summary["recommendations"],
            "sources": link_title
        }

    def _search_ftu(self, question):
        results = self.search_tool.results(question)
        link_title = []
        snippets = ""

        for item in results.get("organic", []):
            url = item.get("link", "")
            if self._is_ftu_domain(url):
                title = item.get("title", "")
                snippet = item.get("snippet", "")

                snippets += f"- {snippet}\n"
                link_title.append({
                    "url": url,
                    "title": title
                })
        
        return snippets, link_title

    def _is_ftu_domain(self, url):
        parsed_url = urlparse(url)
        return ("ftu.edu.vn" in parsed_url.netloc) or ("ForumFTU" in parsed_url.netloc) or ("k64ftubravethestorm" in parsed_url.netloc)


# Khởi tạo
search_agent = SearchAgent()