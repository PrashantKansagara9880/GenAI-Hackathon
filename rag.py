import os
from groq import Groq
from retrival import retrival
from dotenv import load_dotenv
load_dotenv()
from retrival import retrival
client=Groq(api_key=os.getenv("GROQ_API_KEY"))




def mainwork(query,history,session_id):
    
    if history is None:
        history=[]

    contextchunks=retrival(query,session_id)
    context="\n\n".join([
        f"[Source {i}: {doc.metadata.get('source','?')}, page {doc.metadata.get('page','?')}]\n{doc.page_content}"
        for i, doc in enumerate(contextchunks, 1)
    ])

    messages = [
        {
            "role": "system",
            "content": """
            You are DocBuddy, a helpful document assistant.

            Use the retrieved document context as the primary source of truth.

            Use the conversation history to understand follow-up questions,
            references, and pronouns such as:
            - it
            - this
            - that
            - these
            - these two
            - the previous concept

            When answering:
            1. Prefer information from the retrieved documents.
            2. Combine information from multiple retrieved chunks when needed.
            3. Explain concepts clearly and naturally.
            4. If the answer cannot be supported by the retrieved documents, say:
            'I don't have enough information in the uploaded documents to answer that.'
            5. Do not invent facts or citations.
        
            Always end with:

            Sources:
            - source name and page number
            """
        }]
    for msg in history:
        messages.append({
            "role":msg["role"],
            "content":msg["content"]
        })
    messages.append({"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"})

    response=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        stream=True,
        temperature=0
    )
    
    history.append(
        {
            "role":"user",
            "content":query
        })
    history.append(
        {
            "role":"assistant",
            "content":""
        }
    )

    for chunk in response:
        content=chunk.choices[0].delta.content or ""
        history[-1]["content"]+=content
        yield history