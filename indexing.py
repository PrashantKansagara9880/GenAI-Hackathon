from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import gradio as gr

from safety import safe_call
embedding_model=HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device":"cpu"}
)
@safe_call
def indexing(file,session_id,progress=gr.Progress()):
    
    if file is None:
        return "No file uploaded"
    path=file.name
    
    progress(0.05, desc="📄 Loading PDF...")

    loader=PyPDFLoader(path)
    pages=loader.load()
   
    progress(0.15,desc=f"📑 Found {len(pages)} pages...")
    name=os.path.basename(file.name)
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    progress(0.35,desc="✂️ Splitting document into chunks...")
    chunks=splitter.split_documents(pages)
    progress(0.50,desc="🏷️ Preparing metadata...")
    allchunks=[]
    i=0
    for chunk in chunks:
        chunk.metadata["source"]=name
        chunk.metadata["page"]=chunk.metadata.get("page",0)+1
        allchunks.append(chunk) 
        i+=1

    progress(0.65,desc="💾 Saving vectors...")
    vectorstore=Chroma.from_documents(
        documents=allchunks,
        embedding=embedding_model,
        collection_name=session_id,
        persist_directory="./chroma_store"
    )
    progress(1.0,desc="✅ Indexing completed!")
    return f"Indexed {len(allchunks)} chunks into ChromaDB."