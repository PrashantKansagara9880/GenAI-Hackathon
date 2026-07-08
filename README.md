# 👁️ HybridSight

> **One Agent. Three Sources. One Answer.**

HybridSight is a multimodal AI assistant developed for **PS5 – Multimodal Q&A Pro** in the **MSTC GenAI Summer of Code 2026 Hackathon**.

It unifies **Retrieval-Augmented Generation (RAG)**, **Vision Understanding**, and **Live Web Search** behind a **custom reasoning agent** that automatically decides which tool(s) should answer a user's query. Instead of exposing isolated AI capabilities, HybridSight provides a single conversational interface capable of reasoning across multiple information sources while remaining transparent through a reasoning trace.

---

# 🌐 Live Demo

**Hugging Face Space**

> https://huggingface.co/spaces/PrashantKansagara9880/GenAI-Hackathon

---

# 💻 GitHub Repository

> https://github.com/PrashantKansagara9880/GenAI-Hackathon

---

# 📌 Problem Statement

**PS5 – Multimodal Q&A Pro**

Modern AI assistants are often specialized for only one modality—they can either search uploaded documents, analyze images, or retrieve information from the web.

The challenge is to build **one intelligent assistant** capable of:

- Answering questions from uploaded PDF documents
- Understanding uploaded images
- Searching the live web
- Automatically selecting the appropriate tool(s)
- Displaying a transparent reasoning trace
- Deploying as a live web application

HybridSight addresses this by integrating all three capabilities under a single custom reasoning agent.

---

# ✨ Features

## 💬 Hybrid Chat

- Custom reasoning agent
- Automatic tool selection
- Multi-tool reasoning
- Web Search + RAG + Vision
- Collapsible reasoning trace
- Graceful error handling

---

## 📄 Document QA

- PDF upload
- ChromaDB indexing
- Semantic retrieval
- Multi-turn conversations
- Streaming responses
- Source-aware answers

---

## 🖼️ Image Studio

- Image upload
- Adjustable image resolution
- Metadata preview
- Vision-based question answering
- Token-aware image preprocessing
- Automatic image validation

---

# 🏗️ System Architecture

```
                     User
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
   Hybrid Chat    Document QA    Image Studio
        │              │              │
        ▼              ▼              ▼
  Custom Agent      RAG Engine    Vision Pipeline
        │              │              │
        ├──────── search_documents ───┤
        ├────────── web_search ───────┤
        └──────── describe_image ─────┘
                       │
                       ▼
                    Groq API
```

---

# ⚙️ Technology Stack

| Component | Technology |
|------------|------------|
| Language | Python |
| UI | Gradio |
| Reasoning | Custom Planner Agent |
| LLM | Llama 3.3 70B (Groq) |
| Vision | Llama 4 Scout Vision |
| Retrieval | LangChain |
| Vector Database | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 |
| Search | DuckDuckGo Search |
| Image Processing | Pillow |

---

# 📂 Project Structure

```
HybridSight/
│
├── app.py
├── agent.py
├── tools.py
├── rag.py
├── retrival.py
├── indexing.py
├── vision.py
├── safety.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Running Locally

## 1. Clone the repository

```bash
git clone https://github.com/PrashantKansagara9880/GenAI-Hackathon
cd HybridSight
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Create a `.env` file

```env
GROQ_API_KEY=YOUR_API_KEY
```

## 4. Launch

```bash
python app.py
```

---

# 🧪 Tested Scenarios

✔ Pure document retrieval

✔ Pure web search

✔ Pure image understanding

✔ Image + Document reasoning

✔ Graceful API failure handling

✔ Missing document handling

✔ Missing image handling

✔ Session-isolated document retrieval

✔ Progressive document indexing

---

# 🏗️ Design Decisions

## 1. Custom Reasoning Agent

Rather than relying entirely on a prebuilt ReAct implementation, HybridSight uses a lightweight custom reasoning loop.

```
Question
    │
Planner
    │
Tool
    │
Observation
    │
Planner
    │
Final Answer
```

This provides complete control over:

- Tool routing
- Planning logic
- Stopping conditions
- Memory
- Reasoning traces
- Future extensibility

---

## 2. Runtime Image Placeholder

Instead of passing an entire Base64 image through the planner, the planner reasons using only

```
CURRENT_IMAGE
```

During execution the runtime replaces this placeholder with the actual encoded image before invoking the Vision model.

```
Planner
    │
CURRENT_IMAGE
    │
Runtime
    │
Vision Tool
```

### Benefits

- Smaller prompts
- Lower token usage
- Faster planning
- Cleaner reasoning
- No duplicate image serialization

---

## 3. Token-Aware Vision Pipeline

The resolution slider is an engineering optimization rather than a UI enhancement.

Large images generate significantly larger Base64 payloads, increasing:

- latency
- token consumption
- network transfer
- memory usage
- request failure probability

Each uploaded image therefore passes through

```
Upload
   │
Resize
   │
JPEG Compression
   │
Token Estimation
   │
Metadata Generation
   │
Validation
   │
Vision Model
```

Only images satisfying the configured threshold are analyzed.

---

## 4. Dedicated Document QA

Hybrid Chat focuses on multimodal reasoning.

Document QA is optimized solely for retrieval.

```
PDF
 │
Embeddings
 │
ChromaDB
 │
Retriever
 │
LLM
```

Separating these workflows keeps retrieval grounded, efficient, and free from unnecessary planning overhead.

---

## 5. Session Isolation

Every user receives an independent ChromaDB collection.

```
collection_name = session_id
```

This prevents retrieval across user sessions and keeps uploaded documents isolated.

---

## 6. Transparent Reasoning Trace

Every reasoning step records:

- Selected tool
- Tool input
- Tool output

This makes tool selection transparent and greatly simplifies debugging.

---

## 7. Unified Safety Layer

A dedicated safety decorator wraps external API calls and gracefully handles failures including:

- Rate limits
- Invalid API keys
- Network failures
- Timeout errors
- Missing files
- Invalid tool requests
- JSON parsing errors

The application continues running instead of crashing.

---

## 8. Progressive Document Indexing

Document indexing provides live progress updates throughout the pipeline.

```
Loading PDF
      ↓
Splitting
      ↓
Preparing Metadata
      ↓
Generating Embeddings
      ↓
Writing ChromaDB
      ↓
Completed
```

This improves user experience during large document uploads.

---

# 📚 Dataset References

HybridSight does not rely on a fixed dataset.

Instead it operates on:

- User-uploaded PDF documents
- User-uploaded images
- Live DuckDuckGo Search results

Embeddings are generated at runtime using **all-MiniLM-L6-v2**.

---

# 🚀 Deployment

HybridSight is designed for deployment on **Hugging Face Spaces**.

API credentials are securely managed using **Hugging Face Repository Secrets**, ensuring no sensitive keys are stored in source code.

---

# 🔮 Future Improvements

- Multi-document collections
- OCR for scanned PDFs
- Citation highlighting
- Streaming reasoning traces
- Parallel tool execution
- Persistent cloud memory
- Additional knowledge tools (Wikipedia, ArXiv, etc.)

---

