# 👁️ HybridSight

> A multimodal AI assistant that combines Retrieval-Augmented Generation (RAG), Web Search, and Vision into a single application through a custom reasoning agent.

HybridSight is a lightweight GenAI application capable of answering questions from uploaded documents, uploaded images, and the web. Instead of relying on a fixed pipeline, it uses a custom-built reasoning agent that dynamically decides which tool(s) should be used for every query while exposing its reasoning process to the user.

---

## ✨ Features

### 💬 Hybrid Chat
- Free-form conversational interface
- Dynamic tool selection using a custom reasoning agent
- Supports document retrieval, web search, and image understanding
- Multi-tool reasoning for complex queries
- Collapsible reasoning trace for every response

### 📄 Document QA
- PDF upload and indexing into ChromaDB
- Semantic retrieval using sentence embeddings
- Multi-turn document conversations
- Streaming responses
- Source-aware answers

### 🖼️ Image Studio
- Image upload with adjustable resolution
- Vision-based question answering
- Image metadata preview
- Token-aware image preprocessing
- Automatic validation before sending images to the Vision model

---

# 🏗️ Architecture

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
| LLM | Llama 3.3 70B (Groq) |
| Vision Model | Llama 4 Scout Vision |
| Framework | Gradio |
| Retrieval | LangChain |
| Vector Database | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 |
| Search | DuckDuckGo Search |
| Image Processing | Pillow |

---

# 📂 Project Structure

```
HybridSight
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

Clone the repository

```bash
git clone <repository-url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```
GROQ_API_KEY=YOUR_API_KEY
```

Run the application

```bash
python app.py
```

---

# ☁️ Deploying on Hugging Face Spaces

1. Create a **Gradio Space**
2. Upload the project files
3. Add the following Secret

```
GROQ_API_KEY
```

4. Hugging Face installs the dependencies automatically from `requirements.txt`.

---

# 🧪 Tested Scenarios

✅ Pure document questions

✅ Pure web-search questions

✅ Pure image questions

✅ Image + document reasoning

✅ Graceful handling of API failures

✅ Missing document handling

✅ Missing image handling

✅ Session-specific document retrieval

---

# 💡 Engineering Decisions

## Custom Reasoning Agent

Instead of relying on a fixed workflow, HybridSight implements its own reasoning loop.

The agent repeatedly:

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

This provides full control over:

- tool routing
- stopping conditions
- reasoning traces
- memory
- runtime behaviour
- tool-specific execution

It also makes it straightforward to add new tools without changing the overall architecture.

---

## Runtime Image Placeholder

A common challenge with multimodal agents is handling image data efficiently.

Instead of sending the complete Base64 image to the planner model, the planner reasons only with a lightweight placeholder:

```
CURRENT_IMAGE
```

During execution, the runtime replaces this placeholder with the actual image before invoking the vision tool.

```
Planner
    │
CURRENT_IMAGE
    │
Runtime
    │
Vision Tool
```

This approach:

- reduces prompt size
- lowers token usage
- avoids unnecessary serialization
- keeps reasoning independent of image size
- prevents duplicate image processing

---

## Token-Aware Vision Pipeline

The resolution slider is not merely a UI feature.

Large images generate significantly larger Base64 payloads, increasing latency, token consumption, and the likelihood of request failures.

Before an image reaches the Vision model, HybridSight performs:

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

Only images that satisfy the configured token threshold are analyzed.

This protects the application from oversized requests while improving responsiveness.

---

## Dedicated Document QA

Hybrid Chat and Document QA serve different purposes.

Hybrid Chat focuses on open-ended multimodal reasoning.

```
Question
      │
 Agent
      │
 Multiple Tools
      │
 Answer
```

Document QA is optimized specifically for retrieval.

```
PDF
   │
Embedding
   │
ChromaDB
   │
Retriever
   │
LLM
```

Separating these workflows keeps document conversations faster, source-grounded, and free from unnecessary planner overhead.

---

## Session Isolation

Each user receives an independent ChromaDB collection.

```
collection_name = session_id
```

This ensures uploaded documents remain isolated between users and prevents retrieval across sessions.

---

## Reasoning Trace

Every tool invocation is recorded and displayed.

For each reasoning step, the application exposes:

- selected tool
- tool input
- tool output

This makes the decision-making process transparent and simplifies debugging.

---

## Unified Safety Layer

External dependencies are wrapped with a unified safety decorator.

Instead of terminating the application, common failures are handled gracefully, including:

- rate limits
- invalid API keys
- network errors
- JSON parsing failures
- timeout errors
- missing files
- invalid tool requests

This keeps the UI responsive even when external services fail.

---

## Progressive Document Indexing

Document indexing provides real-time progress updates.

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

This improves user feedback, especially for larger documents.

---

# 🔮 Future Improvements

- Multi-document collections
- OCR support for scanned PDFs
- Citation highlighting
- Streaming reasoning traces
- Persistent cloud memory
- Additional knowledge tools (Wikipedia, ArXiv, etc.)
- Parallel tool execution

---

