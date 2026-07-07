from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()
client=Groq(api_key=os.getenv("GROQ_API_KEY"))

embeding_model=HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device":"cpu"}
)

@tool 
def get_current_date():
    """
    Returns today's date in YYYY-MM-DD format.
    Use this when the user asks about today's date,
    or when you need to know the current date before searching.
    """
    return datetime.date.today().isoformat()


search=DuckDuckGoSearchRun(
    description="Search the web for real-time information. "
                "Use for current events, news, live prices, "
                "or anything published after 2024. Input: a search query string. "
                "Do NOT use for definitions, history, or background knowledge."
)





@tool 
def web_search(query: str)-> str:
    """
    Use this tool to search the web for real-time information.
    Input: a search query string.
    Output: a summary of the most relevant search results.
    Use this for current events, news, live prices, or anything published after 2024.
    """
    if query.strip().lower() == "timeout_test":
        raise TimeoutError("Simulated timeout")
    print("Web search called with:", query)
    return search.run(query)





@tool
def search_documents(user_string,session_id):

    print(f"search_documents tool used with: {user_string}")
    """Search the user's uploaded documents for information relevent to the query.Use this when the 
    user asks about content from a PDF they uploaded, or references 'the document', 'my notes',
    or 'the file'.
    Do NOT use this for general knowledge or current events"""
    vectorestore=Chroma(
    collection_name=session_id,
    embedding_function=embeding_model,
    persist_directory="./chroma_store"
    )

    if vectorestore._collection.count()==0:
        return "No documents uploaded yet"
   
    retriver=vectorestore.as_retriever(search_kwargs={"k":5})
    
    contextchunks=retriver.invoke(user_string)
    
    if not contextchunks:
        return "No relevent content found in the uploaded documents."
    
    
    context="\n\n".join([
        f"[Source {i}: {doc.metadata.get('source','?')}, page {doc.metadata.get('page','?')}]\n{doc.page_content}"
        for i, doc in enumerate(contextchunks, 1)
    ])


    return context

@tool
def describe_image(image_data: str) -> str:
    """Describe the content of an image. Use when the user uploads an
    image or asks what's in a picture. Input must be a base64 data URI
    (e.g. 'data:image/jpeg;base64,...')."""
    print(f"describe_image tool used with: {image_data}")
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail."},
                    {"type": "image_url", "image_url": {"url": image_data}},
                ],
            }],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not process the image: {e}"

tools=[get_current_date,web_search,search_documents,describe_image]