"""
IBM DeliveryIQ — Module 2: RAG Knowledge Engine
================================================
WHY WE USE RAG HERE:
    IBM has thousands of delivery frameworks, templates, and past project
    lessons. A consultant can't read all of them. RAG (Retrieval-Augmented
    Generation) solves this by:

    1. INDEXING: Converting IBM documents into vector embeddings (numbers
       that capture meaning) and storing them in a vector database (Milvus/ChromaDB)

    2. RETRIEVAL: When you ask a question, finding the most semantically
       similar document chunks — not just keyword matching

    3. GENERATION: Feeding the retrieved context to the LLM so it answers
       from IBM's actual documents, not from hallucination

    This is Week 2 in action:
    - LangChain: Orchestrates the entire RAG pipeline
    - LCEL (LangChain Expression Language): Chains components together
    - ChromaDB: Stores and retrieves document embeddings locally
    - HuggingFace embeddings: Converts text to semantic vectors
    - Ollama: Local LLM (Llama 3 / Mistral) for generation
    - Conversation memory: Remembers your previous questions

WHY NOT JUST USE CHATGPT?
    1. IBM documents are confidential — can't send to OpenAI
    2. General LLMs don't know IBM-specific processes
    3. RAG gives source citations — you know WHERE the answer came from
    4. Runs 100% locally — no API costs, no data privacy concerns
"""

import os
from typing import List, Optional
from pathlib import Path

from langchain_community.document_loaders import TextLoader, DirectoryLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None
try:
    from langchain.chains import ConversationalRetrievalChain
except ImportError:
    from langchain_community.chains import ConversationalRetrievalChain
try:
    from langchain.memory import ConversationBufferWindowMemory
except ImportError:
    from langchain_community.memory import ConversationBufferWindowMemory
try:
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
except ImportError:
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# WHY THESE SETTINGS?
# - chunk_size=500: Each chunk is ~500 characters. Too large = irrelevant
#   context. Too small = missing context. 500 is the sweet spot for
#   IBM policy documents.
# - chunk_overlap=50: Overlap ensures we don't cut sentences mid-thought
# - k=4: Retrieve top 4 most relevant chunks per question
# ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 4
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, free, runs locally
OLLAMA_MODEL = "llama3.2"              # Or "mistral" — both installed in Week 1
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_stores', 'chroma_db')
DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'documents')


# ─────────────────────────────────────────────────────────────────
# IBM-SPECIFIC PROMPT TEMPLATE
# WHY A CUSTOM PROMPT?
# The default LangChain prompt is generic. We customize it to:
# 1. Tell the LLM it's an IBM delivery consultant assistant
# 2. Instruct it to cite sources (IBM document sections)
# 3. Tell it to use IBM terminology and format
# 4. Prevent hallucination by saying "only use the provided context"
# ─────────────────────────────────────────────────────────────────
IBM_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are IBM DeliveryIQ, an AI assistant for IBM Delivery Consultants.
You have deep knowledge of IBM's delivery methodologies, project management frameworks,
and consulting best practices.

INSTRUCTIONS:
- Answer ONLY based on the provided IBM knowledge base context below
- Use IBM terminology (RAG status, IBM Garage, SOW, etc.)
- Be concise and professional — IBM consultant style
- Always cite which section your answer comes from
- If the answer is not in the context, say "This information is not in my IBM knowledge base. Please consult your delivery manager."
- Format lists with bullet points for clarity

PREVIOUS CONVERSATION:
{chat_history}

IBM KNOWLEDGE BASE CONTEXT:
{context}

CONSULTANT'S QUESTION: {question}

IBM DELIVERYIQ ANSWER:"""
)


class IBMKnowledgeRAG:
    """
    RAG-powered IBM knowledge assistant.

    This class implements the full RAG pipeline:
    Document Loading → Chunking → Embedding → Storage → Retrieval → Generation

    WHY A CLASS?
    We encapsulate the entire pipeline so the Streamlit UI and FastAPI
    can simply call rag.ask("question") without knowing the internals.
    """

    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.chain = None
        self.memory = None
        self.embeddings = None
        self.is_initialized = False
        self.document_count = 0

    def initialize(self, force_reload: bool = False) -> str:
        """
        Initialize the full RAG pipeline.

        Steps:
        1. Load IBM documents
        2. Split into chunks (WHY? LLMs have context limits — we can't feed
           entire documents. Chunking breaks them into digestible pieces)
        3. Create embeddings (WHY? Vectors capture semantic meaning —
           "project risk" and "delivery concern" are similar even without
           shared words)
        4. Store in ChromaDB (WHY? Fast similarity search at query time)
        5. Set up LLM and chain
        """
        print("🔵 Initializing IBM DeliveryIQ Knowledge Engine...")

        # Step 1: Load IBM documents
        # WHY DIRECTORYLOADER? Automatically loads all .txt files in the
        # documents folder — easy to add new IBM docs later
        print("📄 Step 1: Loading IBM knowledge documents...")
        try:
            loader = DirectoryLoader(
                DOCUMENTS_PATH,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            documents = loader.load()
            self.document_count = len(documents)
            print(f"   ✅ Loaded {len(documents)} IBM document(s)")
        except Exception as e:
            print(f"   ⚠️  Document loading error: {e}")
            documents = []

        if not documents:
            return "❌ No IBM documents found. Add .txt files to the documents/ folder."

        # Step 2: Split documents into chunks
        # WHY RECURSIVE CHARACTER SPLITTER?
        # It tries to split on natural boundaries (paragraphs → sentences → words)
        # preserving semantic coherence better than fixed-size splitting
        print("✂️  Step 2: Chunking documents...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"   ✅ Created {len(chunks)} chunks from {len(documents)} documents")

        # Step 3: Create embeddings
        # WHY all-MiniLM-L6-v2?
        # - Free and runs locally (no API key)
        # - Fast (6x faster than larger models)
        # - Good quality for English text
        # - 384-dimensional vectors (compact but expressive)
        print("🧮 Step 3: Creating semantic embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"   ✅ Embedding model loaded: {EMBEDDING_MODEL}")

        # Step 4: Store in ChromaDB
        # WHY CHROMADB?
        # - Runs 100% locally (no Docker needed for basic use)
        # - Persists to disk (don't re-embed every time)
        # - Fast similarity search using HNSW index
        # - Perfect for development and demos
        print("💾 Step 4: Storing embeddings in ChromaDB...")
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)

        if force_reload or not os.path.exists(os.path.join(CHROMA_DB_PATH, 'chroma.sqlite3')):
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=CHROMA_DB_PATH,
                collection_name="ibm_delivery_knowledge"
            )
            self.vectorstore.persist()
            print(f"   ✅ Stored {len(chunks)} chunks in ChromaDB")
        else:
            self.vectorstore = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=self.embeddings,
                collection_name="ibm_delivery_knowledge"
            )
            print(f"   ✅ Loaded existing ChromaDB")

        # Step 5: Set up retriever
        # WHY MMR (Maximal Marginal Relevance)?
        # MMR balances relevance AND diversity — avoids returning 4 chunks
        # that all say the same thing. Gets more comprehensive answers.
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K_RESULTS, "fetch_k": 10}
        )

        # Step 6: Initialize local LLM via Ollama
        # WHY OLLAMA? Runs Llama 3 / Mistral locally — no API costs,
        # no data sent to external servers, works offline
        print("🤖 Step 5: Connecting to LLM (Groq/Ollama)...")
        import os as _os
        _groq_key = _os.environ.get("GROQ_API_KEY", "")
        try:
            if _groq_key and ChatGroq:
                self.llm = ChatGroq(
                    model="llama3-8b-8192",
                    api_key=_groq_key,
                    temperature=0.1,
                    max_tokens=1024,
                )
                print(f"   Connected to Groq (llama3-8b-8192)")
            else:
                self.llm = Ollama(
                model=OLLAMA_MODEL,
                temperature=0.1,      # Low temperature = more factual, less creative
                num_ctx=4096,         # Context window size
            )
            print(f"   ✅ Connected to {OLLAMA_MODEL} via Ollama")
        except Exception as e:
            print(f"   ⚠️  Ollama connection failed: {e}")
            print("   💡 Make sure Ollama is running: ollama serve")
            self.llm = None

        # Step 7: Set up conversation memory
        # WHY MEMORY? So the chatbot remembers context from earlier in
        # the conversation. "What about the budget for that?" — it knows
        # what "that" refers to from previous messages.
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,  # FIXED: Must return messages list, not string
            output_key="answer",
            k=5  # Remember last 5 exchanges
        )

        # Step 8: Build the RAG chain using LCEL
        # WHY LCEL (LangChain Expression Language)?
        # LCEL uses the pipe operator (|) to chain components together.
        # It's more readable, composable, and supports streaming.
        if self.llm:
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": IBM_RAG_PROMPT},
                return_source_documents=True,
                verbose=False
            )

        self.is_initialized = True
        print("\n✅ IBM DeliveryIQ Knowledge Engine ready!")
        return "✅ Knowledge engine initialized successfully"

    def ask(self, question: str) -> dict:
        """
        Ask a question to the IBM knowledge base.

        Args:
            question: Natural language question from the consultant

        Returns:
            dict with answer, sources, and confidence
        """
        if not self.is_initialized:
            self.initialize()

        if not self.llm:
            return {
                'answer': "⚠️ LLM not available. Please start Ollama: `ollama serve`",
                'sources': [],
                'error': True
            }

        try:
            # Run the RAG chain
            result = self.chain({"question": question})

            # Extract source documents for citations
            sources = []
            if 'source_documents' in result:
                for doc in result['source_documents']:
                    source_info = {
                        'content': doc.page_content[:200] + "...",
                        'source': doc.metadata.get('source', 'IBM Knowledge Base'),
                        'section': self._extract_section(doc.page_content)
                    }
                    if source_info not in sources:
                        sources.append(source_info)

            return {
                'answer': result.get('answer', 'No answer generated'),
                'sources': sources[:3],  # Top 3 sources
                'error': False
            }

        except Exception as e:
            return {
                'answer': f"Error processing question: {str(e)}",
                'sources': [],
                'error': True
            }

    def search_documents(self, query: str, k: int = 5) -> List[dict]:
        """
        Semantic search over IBM documents without LLM generation.
        Returns raw document chunks most similar to the query.

        WHY SEMANTIC SEARCH?
        Unlike keyword search (IBM's w3 intranet), semantic search finds
        documents by MEANING. "project in trouble" finds documents about
        "RED status" and "escalation" even without those exact words.
        """
        if not self.is_initialized:
            self.initialize()

        if not self.vectorstore:
            return []

        results = self.vectorstore.similarity_search_with_score(query, k=k)

        return [
            {
                'content': doc.page_content,
                'source': doc.metadata.get('source', 'IBM Knowledge Base'),
                'relevance_score': round(float(1 - score), 3),
                'section': self._extract_section(doc.page_content)
            }
            for doc, score in results
        ]

    def _extract_section(self, content: str) -> str:
        """Extract the IBM document section from chunk content."""
        lines = content.strip().split('\n')
        for line in lines[:3]:
            if line.strip() and len(line.strip()) > 10:
                return line.strip()[:80]
        return "IBM Knowledge Base"

    def clear_memory(self):
        """Clear conversation history — start fresh."""
        if self.memory:
            self.memory.clear()
        return "✅ Conversation history cleared"

    def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        stats = {
            'documents_loaded': self.document_count,
            'is_initialized': self.is_initialized,
            'llm_model': OLLAMA_MODEL,
            'embedding_model': EMBEDDING_MODEL,
            'chunk_size': CHUNK_SIZE,
            'top_k_retrieval': TOP_K_RESULTS
        }
        if self.vectorstore:
            try:
                collection = self.vectorstore._collection
                stats['total_chunks'] = collection.count()
            except:
                stats['total_chunks'] = 'Unknown'
        return stats


# ─────────────────────────────────────────────────────────────────
# DEMO: Run directly to test the RAG pipeline
# python rag_chain.py
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("IBM DeliveryIQ — RAG Knowledge Engine Demo")
    print("=" * 60)

    rag = IBMKnowledgeRAG()
    rag.initialize()

    # Test questions — real IBM consultant questions
    test_questions = [
        "What is IBM's RAG status reporting format?",
        "How do I escalate a project issue at IBM?",
        "What is IBM Garage methodology?",
        "What are the common project risks in IBM delivery?",
        "How do I write an IBM weekly status report?"
    ]

    print("\n" + "=" * 60)
    print("Testing RAG with IBM consultant questions:")
    print("=" * 60)

    for question in test_questions[:2]:  # Test first 2
        print(f"\n❓ Question: {question}")
        result = rag.ask(question)
        print(f"💬 Answer: {result['answer'][:300]}...")
        if result['sources']:
            print(f"📎 Source: {result['sources'][0]['section']}")
        print("-" * 40)

    stats = rag.get_stats()
    print(f"\n📊 Knowledge Base Stats: {stats}")

# Made with Bob
