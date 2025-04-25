import os

import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.docstore.in_memory import InMemoryDocstore

from RAG_w_functionsCalling.core .models import embeddings
from RAG_w_functionsCalling.helpers .data_loader import data_loader


class Retriever:
    def __init__(self, data_path, data_level = "folder"):
        self.data_path = data_path
        self.save_local = os.path.join(os.getcwd(), "data_137", "VectorDB", "IntentOutline", data_path.replace("\\", "/").split("/")[-1])
        self.data_level = data_level
        self.build()

    def build(self):
        if os.path.exists(self.save_local):
            vector_store = FAISS.load_local(self.save_local, embeddings, allow_dangerous_deserialization=True)
            self.retriever = VectorStoreRetriever(vectorstore=vector_store)
        
        else:
            if self.data_level == "folder":
                texts = data_loader.load_folder(self.data_path)
            
            elif self.data_level == "multi_folders":
                texts = data_loader.load_data(self.data_path)

            index = faiss.IndexFlatL2(1024)

            vector_store = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

            vector_store.add_documents(texts)
            vector_store.save_local(self.save_local)
            self.retriever = VectorStoreRetriever(vectorstore=vector_store)
        
        return self

    def re_ingest(self):
        vector_store = FAISS.load_local(self.save_local, embeddings, allow_dangerous_deserialization=True)
        self.retriever = VectorStoreRetriever(vectorstore=vector_store)
        return self