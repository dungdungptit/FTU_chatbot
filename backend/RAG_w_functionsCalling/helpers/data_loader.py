import os

from tqdm import tqdm
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data_path = os.path.join(os.getcwd(), "rag", "data")

class DataLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=256,
            length_function=len,
            is_separator_regex=False,
        )

    def load_file(self, file_path):
        data = []
        loader = TextLoader(file_path = file_path, encoding = "utf-8")
        documents = loader.load()
        for doc in documents:
            data.append(doc.page_content)
        
        texts = self.text_splitter.create_documents(data)
        return texts


    def load_folder(self, folder_path):
        texts = []
        for file in tqdm(os.listdir(folder_path), desc = "Loading data..."):
            if file.endswith(".txt"):
                txt_file_path = os.path.join(folder_path, file)
                texts += self.load_file(txt_file_path)

        return texts
    
    def load_data(self, data_path):
        texts = []
        for folder in os.scandir(data_path):
            if folder.is_dir():
                texts += self.load_folder(folder.path)
        
        return texts
    
data_loader = DataLoader()
