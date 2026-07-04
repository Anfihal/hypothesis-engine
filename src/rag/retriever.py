import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader


class SimpleRAG:
	"""Простой RAG для поиска релевантных фрагментов документов."""
	
	def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
		self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
		self.vectorstore = None
		self.text_splitter = RecursiveCharacterTextSplitter(
			chunk_size=1000,
			chunk_overlap=200,
			separators=["\n\n", "\n", ". ", " ", ""]
		)
	
	def load_from_file(self, file_path: str) -> None:
		"""Загрузка документа из файла (PDF или TXT)."""
		if file_path.endswith('.pdf'):
			loader = PyPDFLoader(file_path)
		elif file_path.endswith('.txt'):
			loader = TextLoader(file_path, encoding='utf-8')
		else:
			raise ValueError(f"Unsupported file format: {file_path}")
		
		documents = loader.load()
		self._process_documents(documents)
	
	def load_from_texts(self, texts: List[str]) -> None:
		"""Загрузка документов из списка строк."""
		from langchain_core.documents import Document
		documents = [Document(page_content=text) for text in texts]
		self._process_documents(documents)
	
	def _process_documents(self, documents: List) -> None:
		"""Разбиение документов на чанки и векторизация."""
		chunks = self.text_splitter.split_documents(documents)
		
		if self.vectorstore is None:
			self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
		else:
			self.vectorstore.add_documents(chunks)
	
	def retrieve(self, query: str, k: int = 5) -> str:
		"""Поиск релевантных фрагментов по запросу."""
		if self.vectorstore is None:
			return ""
		
		docs = self.vectorstore.similarity_search(query, k=k)
		return "\n\n---\n\n".join([doc.page_content for doc in docs])
	
	def save(self, path: str = "faiss_index") -> None:
		"""Сохранение индекса на диск."""
		if self.vectorstore:
			self.vectorstore.save_local(path)
	
	def load(self, path: str = "faiss_index") -> None:
		"""Загрузка индекса с диска."""
		if os.path.exists(path):
			self.vectorstore = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)