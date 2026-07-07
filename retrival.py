from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

from safety import safe_call 

embedding_model=HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device":"cpu"}
)
@safe_call
def retrival(query,session_id):
    vectorestore=Chroma(
        embedding_function=embedding_model,
        collection_name=session_id,
        persist_directory="./chroma_store"
    )
    retriver=vectorestore.as_retriever(search_kwargs={"k":5})
    try:
        context = retriver.invoke(query)
    except Exception:
        return []
    return context