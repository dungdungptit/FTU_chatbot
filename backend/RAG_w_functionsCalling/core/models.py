import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from RAG_w_functionsCalling.core.parser import *
from RAG_w_functionsCalling.core.prompts import *

load_dotenv()

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


def get_chat_completion(task: str, params={}):
    prompt, parser = get_prompt_template(task)
    chain = prompt | llm | parser

    response = chain.invoke(params).dict()
    return response


def get_prompt_template(task):
    if task == "rag":
        parser = rag_parser
        prompt_template = rag_prompt

    elif task == "function_calling":
        parser = function_calling_parser
        prompt_template = function_calling_prompt

    elif task == "search":
        parser = rag_parser
        prompt_template = search_prompt

    elif task == "welcome":
        parser = rag_parser
        prompt_template = welcome_prompt

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template + """{format_instructions}"""),
            ("human", "{question}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt_template, parser
