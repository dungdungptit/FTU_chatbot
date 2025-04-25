from RAG_w_functionsCalling.agents.agents import functions, functions_description
from RAG_w_functionsCalling.core.models import get_chat_completion
from RAG_w_functionsCalling.agents.SearchAgent import SearchAgent

from chain_code import normalize_replace_abbreviation_text

# trả về rỗng
def function_calling(question: str) -> str:
    response = get_chat_completion(
        task="function_calling",
        params={"question": question, "functions_description": functions_description},
    )

    function = response["response"]
    if function not in functions.keys():
        raise ValueError(f"Không tìm thấy function: {response}")

    else:
        print(f"Chọn function: {function}")
        question = normalize_replace_abbreviation_text(question)
        response = (functions[function].invoke(question=question))
    
    if response["response"] == "":
        response = SearchAgent.search_agent.invoke(question = question)
    
    recommendations = response["recommendations"]
    sources = response.get("sources", [
        {
            "title": "Website",
            "url": "http://www.ftu.edu.vn/" 
        },
        {
            "title": "Fanpage Tuyển sinh Trường Đại học Ngoại thương",
            "url": "https://www.facebook.com/TuyensinhFTU"
        },
        {
            "title": "Fanpage Diễn đàn sinh viên trường Đại học Ngoại thương - FTU Forum",
            "url": "https://www.facebook.com/ForumFTU"
        },
        {
            "title": "Group tuyển sinh “K64 FTU (2007) - Brave The Storm!”",
            "url": "https://www.facebook.com/groups/k64ftubravethestorm"
        }
    ])
    return {
        "response": response["response"],
        "recommendations": recommendations,
        "sources": sources
    }

def function_calling_re_ingest() -> None:
    success = True
    for function in functions.values():
        function.retriever.re_ingest()

    return success
