from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import OnlinePDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain import PromptTemplate
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import sys
import os

class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# Directory containing your PDF files
pdf_directory = "pdf"

# List to store all document splits
all_splits = []

# Iterate through all PDF files in the directory
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        
        # Load the PDF
        loader = PyPDFLoader(pdf_path)
        data = loader.load()
        
        # Split the document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        splits = text_splitter.split_documents(data)
        
        # Add the splits to the main list
        all_splits.extend(splits)

# Now all_splits contains the chunks from all PDF files
print(f"Total chunks created: {len(all_splits)}")

with SuppressStdout():
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

while True:
    query = input("\nQuery: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue

    # Prompt
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    llm = Ollama(model="llama3:8b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    result = qa_chain({"query": query})