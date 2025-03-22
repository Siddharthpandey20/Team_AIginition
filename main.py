from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import shutil
import re
from typing import List, Optional, Dict, Any
import threading
import requests
from bs4 import BeautifulSoup
import tempfile
import time
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader, WebBaseLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from datetime import datetime
import traceback
from function import (
    extract_text_from_pdf,
    extract_text_from_csv,
    extract_text_from_audio,
    extract_text_from_image,
    extract_text_auto,
    extract_clean_text,
    extract_text_from_url_simple,      # ✅ Add this!
    extract_text_from_js_rendered_url  # ✅ Optional, fallback method in main.py!
)


# Load environment variables
load_dotenv()

# Configure API keys
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=gemini_api_key)

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)

# Create cache directory if it doesn't exist
cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
os.makedirs(cache_dir, exist_ok=True)

# Initialize embeddings with explicit cache directory
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=cache_dir
)

# Create FastAPI app
app = FastAPI(title="DocuChat API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
#Clear data base after an hour 
import threading

def clear_vector_store():
    """Clears FAISS and BM25 index every 1 hour."""
    global all_documents, bm25_index

    while True:
        print("🕒 Waiting 1 hour before clearing vector database...")
        time.sleep(3600)  # *Wait for 1 hour*
        
        with vector_store_lock:
            print("🧹 Clearing vector database...")
            if os.path.exists(VECTOR_DB_PATH):
                shutil.rmtree(VECTOR_DB_PATH)  # Delete FAISS directory
            os.makedirs(VECTOR_DB_PATH, exist_ok=True)

            # Reset global variables
            all_documents = []
            bm25_index = None

            print("✅ Vector database successfully cleared!")

# Run cleanup in the background
threading.Thread(target=clear_vector_store, daemon=True).start()



# Mount static files directory
app.mount("/static", StaticFiles(directory="."), name="static")

# Update the route handlers
@app.get("/")
async def read_root():
    return FileResponse("index.html")

@app.get("/style.css")
async def get_css():
    return FileResponse("style.css", media_type="text/css")

@app.get("/script.js")
async def get_js():
    return FileResponse("script.js", media_type="application/javascript")

# Add favicon route
@app.get("/favicon.ico")
async def get_favicon():
    return FileResponse("static/favicon.ico", media_type="image/x-icon")

# Create data directory
os.makedirs("data", exist_ok=True)
VECTOR_DB_PATH = "data/vector_db"

# Create a thread lock for vector store access
vector_store_lock = threading.Lock()

class URLInput(BaseModel):
    url: str

class ChatInput(BaseModel):
    question: str

# Initialize or load vector store
def get_vector_store():
    """Load or create FAISS vector store safely."""
    global all_documents, bm25_index

    if not os.path.exists(VECTOR_DB_PATH):
        return FAISS.from_texts(["Placeholder document"], embeddings)

    try:
        vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

        if not all_documents:
            all_documents = list(vector_store.docstore._dict.values())

        update_bm25_index()
        return vector_store

    except Exception as e:
        return FAISS.from_texts(["Placeholder document"], embeddings)


# Global document storage
all_documents = []
tokenized_corpus = []
bm25_index = None

def update_bm25_index():
    """Update the BM25 index with FAISS documents."""
    global bm25_index, tokenized_corpus, all_documents

    if not all_documents:
        return

    tokenized_corpus = [doc.page_content.lower().split() for doc in all_documents]
    bm25_index = BM25Okapi(tokenized_corpus)


# Unified text extraction function
def extract_text_from_source(file_path=None, url=None, file_type=None):
    """Extract text using functions from function.py"""
    try:
        if file_path:
            # Use the auto extractor from function.py
            text = extract_text_auto(file_path=file_path)
        elif url:
            # Use the clean text extractor from function.py
            text = extract_clean_text(url)
        else:
            raise ValueError("Either file_path or url must be provided")

        # Convert extracted text to Document format
        doc = Document(
            page_content=text,
            metadata={"source": file_path or url}
        )

        # Split text using hackathon.py parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        return text_splitter.split_documents([doc])
        
    except Exception as e:
        raise ValueError(f"Error extracting text: {e}")

def add_to_vector_store(documents, source_id):
    """Add documents to FAISS and update BM25 index."""
    global all_documents, bm25_index

    if not documents:
        print("❗ No documents to add to FAISS.")
        return 0

    with vector_store_lock:
        vector_store = get_vector_store()

        # Add metadata before storing
        for doc in documents:
            doc.metadata["source"] = source_id
            doc.metadata["timestamp"] = time.time()

        # Add to FAISS
        vector_store.add_documents(documents)
        vector_store.save_local(VECTOR_DB_PATH)

        # Update all_documents and BM25
        all_documents = list(vector_store.docstore._dict.values())
        update_bm25_index()

        print(f"✅ {len(documents)} documents added to FAISS.")
        print(f"📂 FAISS now contains {len(all_documents)} documents.")
        print(f"📄 BM25 index updated with {len(all_documents)} documents.")

        return len(documents)


# Constants for consistent parameter tuning
CHUNK_SIZE = 600  # Same as hackathon.py
CHUNK_OVERLAP = 100
VECTOR_TOP_K = 10
BM25_WEIGHT = 0.3
VECTOR_WEIGHT = 0.7

# Update text splitter settings to match hackathon.py
def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

def hybrid_search(query, all_splits, vector_store, top_n=10):
    """Enhanced hybrid search with better URL content handling."""
    global bm25_index, tokenized_corpus

    if not all_splits:
        return []
    
    if not vector_store:
        return []

    # Keep this print as it shows retrieved documents
    print(f"🔍 Retrieved documents for query: {query}")

    # Query Expansion
    expanded_query = llm.invoke(f"Expand this search query while maintaining its core meaning: '{query}'")
    expanded_query = expanded_query.content if hasattr(expanded_query, "content") else str(expanded_query)

    query_tokens = expanded_query.lower().split()
    scores = bm25_index.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    
    # Keep this print as it shows document retrieval results
    print(f"📊 Found {len(top_indices)} relevant documents")
    
    candidate_docs = [all_splits[i] for i in top_indices]
    vector_results = vector_store.similarity_search_with_score(expanded_query, k=top_n * 2)
    
    # Keep this print as it shows search results
    print(f"🎯 Retrieved {len(vector_results)} relevant documents")

    # Merge results with enhanced weighting
    weighted_results = {}
    
    for doc, score in vector_results:
        weighted_results[doc.page_content] = {
            "doc": doc,
            "score": score * VECTOR_WEIGHT
        }

    for doc in candidate_docs:
        if doc.page_content in weighted_results:
            weighted_results[doc.page_content]["score"] += BM25_WEIGHT
        else:
            weighted_results[doc.page_content] = {
                "doc": doc,
                "score": BM25_WEIGHT
            }

    # Sort and return results
    final_results = sorted(
        weighted_results.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    result_docs = [item["doc"] for item in final_results[:top_n]]
    
    # Log final results
    print(f"✅ Final hybrid search returned {len(result_docs)} documents")
    return result_docs



from langgraph.graph import MessagesState, StateGraph

graph_builder = StateGraph(MessagesState)

from langchain_core.tools import tool
from typing import Any, Dict

from langchain_core.messages import ToolMessage

from langchain_core.messages import ToolMessage

@tool
def retrieve(query: str):
    """Retrieves most relevant information using hybrid search and re-ranks.

    Args:
        query: The search query to use for retrieval.

    Returns:
        Final ranked response as a ToolMessage.
    """
    global all_documents  

    if not all_documents:
        return ToolMessage(content="No documents available. Please upload a document first.", name="retrieve")

    vector_store = get_vector_store()  
    all_splits = all_documents[:]  

    retrieved_docs = hybrid_search(query, all_splits, vector_store, top_n=10)

    sources = [f"{i+1}. {doc.page_content[:300]}..." for i, doc in enumerate(retrieved_docs)]

    print(f"🔎 Retrieved Docs Count: {len(retrieved_docs)}")

    if not retrieved_docs:
        return ToolMessage(content="No relevant documents found.", name="retrieve")

    # Re-ranking prompt
    rerank_prompt = (
        f"You are retrieving information for this query: '{query}'.\nPrioritize the most detailed and specific response  give higher importance to documents that create a logical, flowing narrative and prioritize documents that reflect the temporal progression of events rather than random moments:\n"
    )
    for doc in retrieved_docs:
        rerank_prompt += f"- {doc.page_content[:300]}...\n"

    ranked_docs = llm.invoke(rerank_prompt)
    ranked_content = ranked_docs.content if hasattr(ranked_docs, "content") else "\n".join(sources)

    return ToolMessage(content=ranked_content, name="retrieve")

from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langgraph.prebuilt import ToolNode


# Update the query_or_respond function
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    
    # Return state with new messages array
    return {"messages": state["messages"] + [response]}

tools = ToolNode([retrieve])

# Update the generate function
def generate(state: MessagesState):
    """Generate final structured answer using retrieved context, with re-ranking like hackathon.py."""
    retrieved_contexts = []

    # Extract retrieved documents from messages
    for message in state["messages"]:
        if isinstance(message, ToolMessage) and message.name == "retrieve":
            retrieved_contexts.append(message.content)

    if not retrieved_contexts:
        retrieved_contexts.append("⚠ No relevant data found in retrieval. Please upload a document.")

    # Extract user query properly
    user_query = next(
        (msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)), ""
    )

    rerank_prompt = (
        f"You are retrieving information for this query: '{user_query}'.\nPrioritize the most detailed and specific response  give higher importance to documents that create a logical, flowing narrative and prioritize documents that reflect the temporal progression of events rather than random moments:\n"
    )

    for i, doc in enumerate(retrieved_contexts):
        rerank_prompt += f"{i+1}. {doc}\n"

    rerank_prompt += "\nNow, provide the top-ranked responses in order of importance:"

    # 🔥 Re-ranking LLM call
    ranked_docs_response = llm.invoke(rerank_prompt)
    ranked_context = ranked_docs_response.content.strip().split("\n") if hasattr(ranked_docs_response, "content") else retrieved_contexts

    # Final LLM structured answer generation
    structured_prompt = (
        "You are an advanced research assistant. Use only the following re-ranked context to answer the query. "
        "Do not generate external information. Provide a structured, step-by-step response.\n\n"
        "*Re-ranked Context:*\n" + "\n".join(ranked_context) +
        "\n\nNow, generate a high-quality, structured answer for this user query:"
    )

    conversation_messages = [
        SystemMessage(content=structured_prompt),
        HumanMessage(content=user_query)
    ]

    print(f"🔍 Sending structured query to LLM: {conversation_messages}")
    response = llm.invoke(conversation_messages)

    final_answer = response.content.strip() if hasattr(response, "content") else "⚠ No valid response generated."

    print(f"✅ LLM Structured Response: {final_answer}")
    return {"messages": [AIMessage(content=final_answer)]}


from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

import threading

# Store file processing status
processing_status = {}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        os.makedirs("uploads", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"uploads/{timestamp}_{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"📂 File {file.filename} saved. Now extracting text...")

        # Mark processing as started
        processing_status[file.filename] = "processing"

        # Run extraction in background
        threading.Thread(target=process_file, args=(file_path, file.filename), daemon=True).start()

        return JSONResponse({
            "status": "success",
            "message": "File uploaded successfully and is being processed.",
            "filename": file.filename
        })

    except Exception as e:
        print(f"❗ Error processing file {file.filename}: {e}")
        raise HTTPException(500, detail=str(e))

import concurrent.futures
import time

def process_file(file_path, filename):
    """Extract text and update vector store in a background thread."""
    global processing_status, all_documents
    try:
        print(f"📂 Processing file: {filename}")
        processing_status[filename] = "processing"

        # Use ThreadPoolExecutor to avoid blocking
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(extract_text_from_source, file_path=file_path)
            try:
                docs = future.result(timeout=60)  # Timeout after 60 seconds
            except concurrent.futures.TimeoutError:
                print(f"❗ Timeout while extracting text from {filename}")
                processing_status[filename] = "failed"
                return

        if not docs:
            print(f"❗ No valid text extracted from {filename}")
            processing_status[filename] = "failed"
            return
        
        # Process and chunk the extracted text
        processed_docs = process_extracted_text(docs[0].page_content)
        
        if not processed_docs:
            print(f"❗ No valid content chunks were generated from {filename}")
            processing_status[filename] = "failed"
            return
        
        doc_count = add_to_vector_store(processed_docs, source_id=filename)
        print(f"✅ File {filename} processed successfully. {doc_count} documents added.")

        # Ensure FAISS index is properly saved
        vector_store = get_vector_store()
        vector_store.save_local(VECTOR_DB_PATH)

        # Update all_documents
        all_documents = list(vector_store.docstore._dict.values())

        print(f"📁 Total documents in FAISS after update: {len(all_documents)}")
        processing_status[filename] = "completed"

    except Exception as e:
        print(f"❗ Error processing {filename}: {e}")
        processing_status[filename] = "failed"

@app.get("/processing-status")
async def get_processing_status(filename: str):
    """Check if a file has finished processing."""
    status = processing_status.get(filename, "unknown")
    
    # If failed, return an error message
    if status == "failed":
        return JSONResponse({"status": "failed", "message": "File processing failed."})

    return JSONResponse({"status": status})


@app.get("/list-uploads")
async def list_uploads():
    """List all files in the uploads directory."""
    try:
        uploads_dir = "uploads"
        if not os.path.exists(uploads_dir):
            return JSONResponse([])
            
        files = []
        for file in os.listdir(uploads_dir):
            if os.path.isfile(os.path.join(uploads_dir, file)):
                files.append(file)
        return JSONResponse(files)
    except Exception as e:
        print(f"❗ Error listing uploads: {e}")
        return JSONResponse([])

# Update the chat endpoint
import json

@app.post("/chat/{query}")
async def chat(query: str):
    """Processes user query, retrieves documents, and invokes the response generation pipeline."""
    try:
        print(f"📩 Received query: {query}")

        # Load FAISS and BM25
        vector_store = get_vector_store()

        if not all_documents or len(all_documents) == 0:
            print("⚠ No documents available for search (all_documents is empty).")
            return JSONResponse({"answer": "No relevant documents found. Please upload a document first.", "sources": ""})

        print(f"✅ Running hybrid search for query: {query}")
        results = hybrid_search(query, all_documents, vector_store, top_n=5)

        if not results:
            print("⚠ No relevant information found.")
            return JSONResponse({"answer": "No relevant information found in uploaded documents.", "sources": ""})

        # Debug print preview of retrieved documents
        for i, doc in enumerate(results):
            print(f"📖 Retrieved Doc {i+1}: {doc.page_content[:200]}...")

        from langgraph.graph import MessagesState, StateGraph
        from langgraph.graph import END
        from langgraph.prebuilt import ToolNode, tools_condition
        from langgraph.checkpoint.memory import MemorySaver

        # Ensure LLM is invoked only inside generate()
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node(query_or_respond)
        graph_builder.add_node(tools)
        graph_builder.add_node(generate)

        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory)
        config = {"configurable": {"thread_id": "abc123"}}

        Ans = {"answer": "", "sources": ""}
        print("🚀 Running LangGraph streaming process...")

        for step in graph.stream({"messages": [HumanMessage(content=query)]}, stream_mode="values", config=config):
            if isinstance(step, dict):
                if "tool_message" in step and step["tool_message"].get("name") == "retrieve":
                    retrieved_context = step["tool_message"].get("response", "")
                    Ans["sources"] = retrieved_context

                if "messages" in step and isinstance(step["messages"], list):
                    last_message = step["messages"][-1]
                    
                    if isinstance(last_message, AIMessage):
                        generated_response = last_message.content.strip()
                        if generated_response:
                            Ans["answer"] = generated_response

        if not Ans["answer"]:
            Ans["answer"] = "⚠ I'm unable to find an answer. Please try rephrasing your query."

        print(f"✅ Response generated: {Ans['answer'][:200]}...")
        return JSONResponse(Ans)

    except Exception as e:
        print(f"❗ Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def process_extracted_text(text: str) -> List[Document]:
    """Process extracted text into properly chunked documents."""
    # Check if this looks like CSV data
    if "Dataset Content:" in text:
        # Handle CSV content differently - keep it together
        return [Document(
            page_content=text,
            metadata={
                "source": "csv_data",
                "type": "structured_data",
                "timestamp": datetime.now().isoformat(),
            }
        )]
    
    # Create a text splitter with specific separators for news/article content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=[
            "\n\n",  # Major section breaks
            "\nRelated Story:",  # News article related stories
            "\n---",  # Page breaks
            "\n",    # Regular line breaks
            ". ",    # Sentences
            "? ",    # Questions
            "! ",    # Exclamations
            ", ",    # Clauses
            " ",     # Words
            ""       # Characters
        ],
        length_function=len
    )
    
    # Clean and structure the text
    cleaned_text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    cleaned_text = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\2', cleaned_text)  # Add breaks at sentences
    
    # Create initial document
    doc = Document(
        page_content=cleaned_text,
        metadata={
            "source": "extracted_content",
            "timestamp": datetime.now().isoformat(),
        }
    )
    
    # Split into chunks
    docs = text_splitter.split_documents([doc])
    
    # Add additional metadata and filtering
    processed_docs = []
    for i, doc in enumerate(docs):
        # Skip chunks that are too short or appear to be navigation/footer content
        if len(doc.page_content.strip()) < 50 or re.search(r'(privacy policy|terms of use|©|all rights reserved)', doc.page_content.lower()):
            continue
            
        doc.metadata.update({
            "chunk_id": i,
            "chunk_type": "content",
        })
        processed_docs.append(doc)
    
    return processed_docs

# Update file processing in upload endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        os.makedirs("uploads", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"uploads/{timestamp}_{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"📂 File {file.filename} saved. Now extracting text...")

        # Mark processing as started
        processing_status[file.filename] = "processing"

        # Run extraction in background
        threading.Thread(target=process_file, args=(file_path, file.filename), daemon=True).start()

        return JSONResponse({
            "status": "success",
            "message": "File uploaded successfully and is being processed.",
            "filename": file.filename
        })

    except Exception as e:
        print(f"❗ Error processing file {file.filename}: {e}")
        raise HTTPException(500, detail=str(e))

import concurrent.futures
import time

def process_file(file_path, filename):
    """Extract text and update vector store in a background thread."""
    global processing_status, all_documents
    try:
        print(f"📂 Processing file: {filename}")
        processing_status[filename] = "processing"

        # Use ThreadPoolExecutor to avoid blocking
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(extract_text_from_source, file_path=file_path)
            try:
                docs = future.result(timeout=60)  # Timeout after 60 seconds
            except concurrent.futures.TimeoutError:
                print(f"❗ Timeout while extracting text from {filename}")
                processing_status[filename] = "failed"
                return

        if not docs:
            print(f"❗ No valid text extracted from {filename}")
            processing_status[filename] = "failed"
            return
        
        # Process and chunk the extracted text
        processed_docs = process_extracted_text(docs[0].page_content)
        
        if not processed_docs:
            print(f"❗ No valid content chunks were generated from {filename}")
            processing_status[filename] = "failed"
            return
        
        doc_count = add_to_vector_store(processed_docs, source_id=filename)
        print(f"✅ File {filename} processed successfully. {doc_count} documents added.")

        # Ensure FAISS index is properly saved
        vector_store = get_vector_store()
        vector_store.save_local(VECTOR_DB_PATH)

        # Update all_documents
        all_documents = list(vector_store.docstore._dict.values())

        print(f"📁 Total documents in FAISS after update: {len(all_documents)}")
        processing_status[filename] = "completed"

    except Exception as e:
        print(f"❗ Error processing {filename}: {e}")
        processing_status[filename] = "failed"

@app.post("/upload-url")
async def upload_url(url_input: URLInput):
    """Handle URL uploads."""
    try:
        url = url_input.url
        print(f"🌐 Processing URL: {url}")
        
        # Mark processing as started
        processing_status[url] = "processing"

        # Run URL processing in background
        threading.Thread(target=process_url, args=(url,), daemon=True).start()

        return JSONResponse({
            "status": "success",
            "message": "URL is being processed.",
            "filename": url
        })

    except Exception as e:
        print(f"❗ Error processing URL {url}: {e}")
        raise HTTPException(500, detail=str(e))

def process_url(url: str):
    """Process URL and add to vector store in background thread."""
    global processing_status, all_documents
    try:
        print(f"🌐 Processing URL: {url}")
        processing_status[url] = "processing"

        # Extract text from URL
        extracted_text = extract_text_from_url_simple(url)  # Use simple URL extraction first
        if not extracted_text or "❗ Error" in extracted_text:
            # If simple extraction fails, try JS-rendered version
            extracted_text = extract_text_from_js_rendered_url(url)

        if not extracted_text or "❗ Error" in extracted_text:
            print(f"❗ Failed to extract text from URL: {url}")
            processing_status[url] = "failed"
            return

        # Create initial document
        doc = Document(
            page_content=extracted_text,
            metadata={
                "source": url,
                "type": "url",
                "timestamp": datetime.now().isoformat()
            }
        )

        # Process and chunk the text
        processed_docs = process_extracted_text(doc.page_content)
        
        if not processed_docs:
            print(f"❗ No valid content chunks generated from URL: {url}")
            processing_status[url] = "failed"
            return

        # Add source URL to all chunks
        for doc in processed_docs:
            doc.metadata["url"] = url
            doc.metadata["type"] = "url"

        # Add to vector store
        doc_count = add_to_vector_store(processed_docs, source_id=url)
        print(f"✅ URL {url} processed successfully. Added {doc_count} chunks to vector store")

        # Update vector store
        vector_store = get_vector_store()
        vector_store.save_local(VECTOR_DB_PATH)

        # Update global documents
        all_documents = list(vector_store.docstore._dict.values())
        update_bm25_index()

        print(f"📁 Vector store now contains {len(all_documents)} total documents")
        processing_status[url] = "completed"

    except Exception as e:
        print(f"❗ Error processing URL {url}: {str(e)}")
        traceback.print_exc()
        processing_status[url]="failed"
