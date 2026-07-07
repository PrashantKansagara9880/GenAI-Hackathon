from agent import Agent
from indexing import indexing
import uuid
import gradio as gr
from langchain_groq import ChatGroq
from safety import safe_call
from vision import image_to_data_uri,ask_vision
import os
from dotenv import load_dotenv
from tools import tools
from rag import mainwork
load_dotenv()
llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)
agent=Agent(llm,tools,12)
os.makedirs("chroma_store", exist_ok=True)

if not os.path.exists("memory.json"):
    with open("memory.json", "w") as f:
        f.write("{}")

@safe_call
def handle_image(image_path, resolution):

    if image_path is None:
        return (
            None,  
            """
### 📊 Image Metadata

⚠️ No image uploaded.

Please upload an image first.
""",
            False  # image_ready
        )
    result = image_to_data_uri(
        image_path,
        resolution
    )

    

    metadata = result["metadata"]

    metadata_md = f"""
### 📊 Image Metadata

**Original Resolution**
{metadata["original_width"]} × {metadata["original_height"]}

**Current Resolution**
{metadata["new_width"]} × {metadata["new_height"]}

**Resolution**
{metadata["resolution_percent"]}%

**Image Size**
{metadata["image_size_kb"]} KB

**Estimated Tokens**
{metadata["estimated_tokens"]}

**Recommended**
{"✅ Yes" if metadata["recommended"] else "⚠️ Reduce Resolution"}
"""

    if metadata["recommended"]:
        agent.current_image=result["data_uri"]
        return result["data_uri"], metadata_md, metadata["recommended"]
    
    return None, metadata_md, metadata["recommended"]




@safe_call
def chat(message,history,session_id):
    
    final_answer,trace_log=agent.run(message,session_id)
        
    if history is None:
        history=[]
    
    history.append(
    {
        "role": "user",
        "content": message
    })

    history.append(
        {
            "role": "assistant",
            "content": final_answer
        }
    )

    return history, session_id, trace_log or "No tools were called."

with gr.Blocks() as demo:

    session_id=gr.State(value=lambda: str(uuid.uuid4()))
    image_uri_state = gr.State(value=None)
    image_ready = gr.State(value=False)
    gr.Markdown("# 👁️ HybridSight — RAG + Web + Vision Agent")
        
    with gr.Tab("💬 Hybrid Chat"):

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=440, label="Conversation")
                msg_box = gr.Textbox(placeholder="Ask anything…", show_label=False)
                submit_btn = gr.Button("Send", variant="primary")
                
            
            with gr.Column(scale=1):
                with gr.Accordion("Reasoning Trace", open=False):
                    trace_box = gr.Textbox(label="Tools called during last response",lines=14, interactive=False)
            
            submit_btn.click(
                    chat,
                    inputs=[msg_box, chatbot, session_id],
                    outputs=[chatbot, session_id,trace_box]
                ).then(lambda: (""),outputs=[msg_box])
            msg_box.submit(
                chat,
                inputs=[msg_box, chatbot, session_id],
                outputs=[chatbot, session_id,trace_box]
            ).then(lambda: (""),outputs=[msg_box])


    with gr.Tab("📄 Document QA"):
        with gr.Row():
            with gr.Column():
                pdf_upload = gr.File(label="Upload a PDF", file_types=[".pdf"])
            with gr.Column():
                index_status = gr.Textbox(label="Indexing status", interactive=False)
                index_btn = gr.Button("📥 Index Document", variant="primary")

        index_btn.click(
    indexing,
    inputs=[pdf_upload, session_id],
    outputs=[index_status]
).then(lambda: None, outputs=[pdf_upload])
        doc_chatbot=gr.Chatbot(height=380)
        doc_input=gr.Textbox(placeholder="Ask about the document…", show_label=False)
        doc_submit_btn=gr.Button("Ask", variant="primary")
        doc_submit_btn.click(
            mainwork,
            inputs=[doc_input,doc_chatbot,session_id],
            outputs=[doc_chatbot]
        ).then(lambda: (""),outputs=[doc_input])
        doc_input.submit(
            mainwork,
            inputs=[doc_input,doc_chatbot,session_id],
            outputs=[doc_chatbot]
        ).then(lambda: (""),outputs=[doc_input])
        
    
    with gr.Tab("🖼️ Image Studio"):

        gr.Markdown("## 🖼️ Image Studio")
        gr.Markdown("Upload an image, adjust its resolution, and ask questions about it.")
        with gr.Row():
            with gr.Column(scale=1):
                image_upload = gr.Image(label="Upload Image",type="filepath")

                resolution_slider = gr.Slider(minimum=10,maximum=100,value=50,step=5,label="Image Resolution (%)")

                metadata_box = gr.Markdown("""
### 📊 Image Metadata

No image uploaded.
""")

                analyse_btn = gr.Button(
                    "🔍 Analyse Image",
                    variant="primary"
                )

            with gr.Column(scale=2):

                image_question = gr.Textbox(
                placeholder="Ask anything about the uploaded image...",
                label="Question"
            )
                ask_btn = gr.Button("Ask")
                image_answer = gr.Textbox(
                label="Vision Response",
                lines=12,
                interactive=False
            )
        
        analyse_btn.click(
        handle_image,
        inputs=[image_upload,resolution_slider],
        outputs=[image_uri_state,metadata_box,image_ready])
        ask_btn.click(
            ask_vision,
            inputs=[image_uri_state,image_ready,image_question],
            outputs=[image_answer]
        )
        image_question.submit(
            ask_vision,
            inputs=[image_uri_state,image_ready,image_question],
            outputs=[image_answer]
        )
demo.queue()
demo.launch()
