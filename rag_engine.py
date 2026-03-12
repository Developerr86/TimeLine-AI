"""
RAG (Retrieval-Augmented Generation) Engine for Timeline-AI.
Provides local semantic search and chat capabilities over captured notes.

Uses:
- ChromaDB for vector storage (persistent, local)
- SentenceTransformers for embeddings (all-MiniLM-L6-v2)
- Ollama for local LLM inference
"""

import re
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

# Vector store directory
VECTOR_STORE_DIR = Path(__file__).parent / "media" / "vector_store"

# Media directories for frame data
MEDIA_VIDEO_DIR = Path(__file__).parent / "media" / "video"


class RAGEngine:
    """
    Local RAG engine for semantic search and chat over captured notes.
    
    Chunking Strategy:
    - Target size: 250-400 tokens (~1000-1600 characters)
    - Strict sentence boundaries (never cut a sentence)
    - Attempts to keep paragraphs together for coherence
    
    Retrieval:
    - Top-k = 5 chunks
    - Context limit: ~1500 tokens total
    """
    
    # Chunking parameters
    TARGET_CHUNK_TOKENS = 350  # Target ~350 tokens per chunk
    MAX_CHUNK_TOKENS = 400     # Hard limit
    MIN_CHUNK_TOKENS = 100    # Minimum viable chunk
    CHARS_PER_TOKEN = 4        # Approximation: 1 token ≈ 4 chars
    
    # Retrieval parameters
    TOP_K = 5                  # Number of chunks to retrieve
    MAX_CONTEXT_TOKENS = 1500  # Maximum context for LLM prompt
    
    def __init__(self, ollama_model: str = "qwen3.5:2b"):
        """
        Initialize the RAG engine.
        
        Args:
            ollama_model: Name of the Ollama model to use for chat
        """
        self.ollama_model = ollama_model
        self._embedding_model: Optional[SentenceTransformer] = None
        self._chroma_client = None
        self._collection = None
        
        # Ensure vector store directory exists
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Lazy initialization flags
        self._initialized = False
    
    def _ensure_initialized(self) -> bool:
        """Lazy initialization of models and database."""
        if self._initialized:
            return True
        
        if not HAS_CHROMADB:
            print("❌ ChromaDB not installed. Run: pip install chromadb")
            return False
        
        if not HAS_SENTENCE_TRANSFORMERS:
            print("❌ SentenceTransformers not installed. Run: pip install sentence-transformers")
            return False
        
        try:
            # Initialize embedding model
            print("🔄 Loading embedding model (all-MiniLM-L6-v2)...")
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded")
            
            # Initialize ChromaDB with persistent storage
            print(f"🔄 Initializing ChromaDB at {VECTOR_STORE_DIR}...")
            self._chroma_client = chromadb.PersistentClient(
                path=str(VECTOR_STORE_DIR),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self._collection = self._chroma_client.get_or_create_collection(
                name="timeline_notes",
                metadata={"description": "Timeline-AI captured notes"}
            )
            print(f"✅ ChromaDB initialized (collection: timeline_notes, docs: {self._collection.count()})")
            
            self._initialized = True
            return True
            
        except Exception as e:
            print(f"❌ RAG initialization failed: {e}")
            return False
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text (1 token ≈ 4 chars)."""
        return len(text) // self.CHARS_PER_TOKEN
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using robust regex.
        Handles common abbreviations and edge cases.
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Common abbreviations that shouldn't trigger sentence breaks
        abbreviations = ['Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sr', 'Jr', 'vs', 'etc', 'i.e', 'e.g']
        
        # Temporarily replace abbreviations with placeholders
        placeholders = {}
        for i, abbr in enumerate(abbreviations):
            placeholder = f'__ABBR{i}__'
            # Match abbreviation followed by period
            pattern = re.escape(abbr) + r'\.'
            text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
            placeholders[placeholder] = abbr + '.'
        
        # Now split on sentence boundaries: .!? followed by space and capital letter
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Restore abbreviations
        restored_sentences = []
        for sent in sentences:
            for placeholder, original in placeholders.items():
                sent = sent.replace(placeholder, original)
            restored_sentences.append(sent)
        
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in restored_sentences if s.strip()]
        
        return sentences
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs (double newline or multiple breaks)."""
        paragraphs = re.split(r'\n\s*\n+', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks suitable for embedding.
        
        Strategy:
        1. Split into paragraphs first
        2. For each paragraph, accumulate sentences until ~350 tokens
        3. Never cut a sentence in half
        4. If a paragraph is too long, split at sentence boundaries
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        # Split into paragraphs first
        paragraphs = self._split_into_paragraphs(text)
        
        for paragraph in paragraphs:
            para_tokens = self._estimate_tokens(paragraph)
            
            # If paragraph fits in current chunk with room to spare
            if current_tokens + para_tokens <= self.TARGET_CHUNK_TOKENS:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_tokens += para_tokens
                continue
            
            # If current chunk is substantial, save it and start new
            if current_tokens >= self.MIN_CHUNK_TOKENS:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_tokens = 0
            
            # If paragraph itself is within limits, add it as/to chunk
            if para_tokens <= self.MAX_CHUNK_TOKENS:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                    current_tokens += para_tokens
                else:
                    current_chunk = paragraph
                    current_tokens = para_tokens
            else:
                # Paragraph is too large - split by sentences
                sentences = self._split_into_sentences(paragraph)
                
                for sentence in sentences:
                    sent_tokens = self._estimate_tokens(sentence)
                    
                    # If adding sentence exceeds limit
                    if current_tokens + sent_tokens > self.MAX_CHUNK_TOKENS:
                        if current_tokens >= self.MIN_CHUNK_TOKENS:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                        current_tokens = sent_tokens
                    else:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                        current_tokens += sent_tokens
        
        # Don't forget the last chunk
        if current_chunk and current_tokens >= self.MIN_CHUNK_TOKENS:
            chunks.append(current_chunk.strip())
        elif current_chunk:
            # If last chunk is too small, append to previous if possible
            if chunks:
                last_chunk = chunks[-1]
                combined_tokens = self._estimate_tokens(last_chunk + "\n\n" + current_chunk)
                if combined_tokens <= self.MAX_CHUNK_TOKENS * 1.2:  # Allow slight overflow
                    chunks[-1] = last_chunk + "\n\n" + current_chunk
                else:
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(current_chunk.strip())
        
        return chunks
    
    def index_session(self, session_id: str, text: str, 
                      source: str = "unknown", title: str = None) -> Dict[str, Any]:
        """
        Index a session's text content for semantic search.
        
        Args:
            session_id: The capture session ID
            text: The text content to index
            source: Source type ("video", "doc", "web")
            title: Optional title for the content
            
        Returns:
            Dict with indexing status and chunk count
        """
        result = {
            "success": False,
            "session_id": session_id,
            "chunks_indexed": 0,
            "errors": []
        }
        
        if not self._ensure_initialized():
            result["errors"].append("RAG engine not initialized")
            return result
        
        if not text or not text.strip():
            result["errors"].append("No text content to index")
            return result
        
        try:
            # Chunk the text
            chunks = self.chunk_text(text)
            
            if not chunks:
                result["errors"].append("Text too short to chunk")
                return result
            
            print(f"📚 Indexing session {session_id}: {len(chunks)} chunks")
            
            # Generate embeddings
            embeddings = self._embedding_model.encode(chunks, show_progress_bar=False)
            
            # Prepare documents for ChromaDB
            ids = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                # Generate unique ID for each chunk
                chunk_hash = hashlib.md5(f"{session_id}_{i}_{chunk[:50]}".encode()).hexdigest()[:12]
                chunk_id = f"{session_id}_{i}_{chunk_hash}"
                
                ids.append(chunk_id)
                metadatas.append({
                    "session_id": session_id,
                    "source": source,
                    "title": title or "Untitled",
                    "chunk_index": i,
                    "indexed_at": datetime.utcnow().isoformat()
                })
            
            # Delete existing chunks for this session (update scenario)
            existing = self._collection.get(
                where={"session_id": session_id}
            )
            if existing and existing.get("ids"):
                self._collection.delete(ids=existing["ids"])
                print(f"  🔄 Removed {len(existing['ids'])} existing chunks for session")
            
            # Add new chunks
            self._collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=chunks,
                metadatas=metadatas
            )
            
            result["success"] = True
            result["chunks_indexed"] = len(chunks)
            print(f"  ✅ Indexed {len(chunks)} chunks for session {session_id}")
            
        except Exception as e:
            result["errors"].append(f"Indexing failed: {str(e)}")
            print(f"  ❌ Indexing error: {e}")
        
        return result
    
    def query(self, user_query: str, top_k: int = None) -> Dict[str, Any]:
        """
        Query the indexed notes and generate a response using local LLM.
        
        Args:
            user_query: The user's question
            top_k: Number of chunks to retrieve (default: 5)
            
        Returns:
            Dict with response, sources, and metadata
        """
        result = {
            "success": False,
            "response": None,
            "sources": [],
            "chunks_used": 0,
            "errors": []
        }
        
        if not self._ensure_initialized():
            result["errors"].append("RAG engine not initialized")
            return result
        
        if not user_query or not user_query.strip():
            result["errors"].append("Empty query")
            return result
        
        if not HAS_OLLAMA:
            result["errors"].append("Ollama not installed. Run: pip install ollama")
            return result
        
        top_k = top_k or self.TOP_K
        
        try:
            # Check if we have any documents
            if self._collection.count() == 0:
                result["response"] = "I don't have any notes indexed yet. Capture some content first using the Video, Document, or Web capture features."
                result["success"] = True
                return result
            
            # Embed the query
            query_embedding = self._embedding_model.encode([user_query])[0]
            
            # Search ChromaDB
            search_results = self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            if not search_results or not search_results.get("documents") or not search_results["documents"][0]:
                result["response"] = "I couldn't find any relevant information in your notes for this query."
                result["success"] = True
                return result
            
            # Extract results
            documents = search_results["documents"][0]
            metadatas = search_results["metadatas"][0] if search_results.get("metadatas") else []
            distances = search_results["distances"][0] if search_results.get("distances") else []
            
            # Build context string, respecting token limit
            context_parts = []
            total_tokens = 0
            sources_set = set()
            
            for i, doc in enumerate(documents):
                doc_tokens = self._estimate_tokens(doc)
                
                if total_tokens + doc_tokens > self.MAX_CONTEXT_TOKENS:
                    break
                
                context_parts.append(doc)
                total_tokens += doc_tokens
                
                # Track sources
                if i < len(metadatas):
                    meta = metadatas[i]
                    source_info = {
                        "session_id": meta.get("session_id", "unknown"),
                        "source": meta.get("source", "unknown"),
                        "title": meta.get("title", "Untitled"),
                        "relevance": 1 - distances[i] if i < len(distances) else None
                    }
                    source_key = source_info["session_id"]
                    if source_key not in sources_set:
                        sources_set.add(source_key)
                        result["sources"].append(source_info)
            
            context_string = "\n\n---\n\n".join(context_parts)
            result["chunks_used"] = len(context_parts)
            
            # Build prompt
            system_prompt = """You are a helpful study assistant. Answer the user's question based ONLY on the context provided below. 
If the context doesn't contain enough information to answer the question, say so honestly.
Be concise but thorough. Use bullet points or numbered lists when appropriate."""

            user_prompt = f"""Context:
{context_string}

Question: {user_query}"""

            # Call Ollama
            print(f"🤖 Querying Ollama ({self.ollama_model})...")
            
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            result["response"] = response["message"]["content"]
            result["success"] = True
            print(f"  ✅ Generated response ({len(result['response'])} chars)")
            
        except Exception as e:
            error_msg = str(e)
            if "connection refused" in error_msg.lower() or "connect" in error_msg.lower():
                result["errors"].append("Ollama is not running. Start it with: ollama serve")
            else:
                result["errors"].append(f"Query failed: {error_msg}")
            print(f"  ❌ Query error: {e}")
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed content."""
        if not self._ensure_initialized():
            return {
                "initialized": False,
                "total_chunks": 0,
                "sessions": []
            }
        
        try:
            total_count = self._collection.count()
            
            # Get unique sessions
            if total_count > 0:
                all_docs = self._collection.get(include=["metadatas"])
                sessions = {}
                for meta in all_docs.get("metadatas", []):
                    sid = meta.get("session_id", "unknown")
                    if sid not in sessions:
                        sessions[sid] = {
                            "session_id": sid,
                            "source": meta.get("source", "unknown"),
                            "title": meta.get("title", "Untitled"),
                            "chunk_count": 0
                        }
                    sessions[sid]["chunk_count"] += 1
                
                session_list = list(sessions.values())
            else:
                session_list = []
            
            return {
                "initialized": True,
                "total_chunks": total_count,
                "sessions": session_list,
                "embedding_model": "all-MiniLM-L6-v2",
                "llm_model": self.ollama_model
            }
        except Exception as e:
            return {
                "initialized": True,
                "total_chunks": 0,
                "sessions": [],
                "error": str(e)
            }
    
    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Delete all indexed chunks for a session."""
        result = {
            "success": False,
            "deleted_count": 0,
            "errors": []
        }
        
        if not self._ensure_initialized():
            result["errors"].append("RAG engine not initialized")
            return result
        
        try:
            existing = self._collection.get(
                where={"session_id": session_id}
            )
            
            if existing and existing.get("ids"):
                self._collection.delete(ids=existing["ids"])
                result["deleted_count"] = len(existing["ids"])
                result["success"] = True
                print(f"🗑️ Deleted {result['deleted_count']} chunks for session {session_id}")
            else:
                result["success"] = True
                print(f"ℹ️ No chunks found for session {session_id}")
                
        except Exception as e:
            result["errors"].append(f"Delete failed: {str(e)}")
            print(f"❌ Delete error: {e}")
        
        return result
    
    def get_chunks_for_sessions(self, session_ids: List[str]) -> Dict[str, Any]:
        """
        Retrieve all chunks for the specified session IDs.
        
        Args:
            session_ids: List of session IDs to retrieve chunks for
            
        Returns:
            Dict with chunks, metadata, and session info
        """
        result = {
            "success": False,
            "chunks": [],
            "sessions_found": [],
            "total_chunks": 0,
            "errors": []
        }
        
        if not self._ensure_initialized():
            result["errors"].append("RAG engine not initialized")
            return result
        
        if not session_ids:
            result["errors"].append("No session IDs provided")
            return result
        
        try:
            all_chunks = []
            sessions_found = set()
            
            for session_id in session_ids:
                # Get chunks for this session
                session_docs = self._collection.get(
                    where={"session_id": session_id},
                    include=["documents", "metadatas"]
                )
                
                if session_docs and session_docs.get("documents"):
                    documents = session_docs["documents"]
                    metadatas = session_docs.get("metadatas", [])
                    
                    for i, doc in enumerate(documents):
                        chunk_info = {
                            "text": doc,
                            "session_id": session_id,
                            "title": metadatas[i].get("title", "Untitled") if i < len(metadatas) else "Untitled",
                            "source": metadatas[i].get("source", "unknown") if i < len(metadatas) else "unknown",
                            "chunk_index": metadatas[i].get("chunk_index", i) if i < len(metadatas) else i
                        }
                        all_chunks.append(chunk_info)
                        sessions_found.add(session_id)
            
            # Sort chunks by session and chunk index
            all_chunks.sort(key=lambda x: (x["session_id"], x["chunk_index"]))
            
            result["success"] = True
            result["chunks"] = all_chunks
            result["sessions_found"] = list(sessions_found)
            result["total_chunks"] = len(all_chunks)
            print(f"📚 Retrieved {len(all_chunks)} chunks from {len(sessions_found)} sessions")
            
        except Exception as e:
            result["errors"].append(f"Chunk retrieval failed: {str(e)}")
            print(f"❌ Chunk retrieval error: {e}")
        
        return result
    
    def generate_notes(self, session_ids: List[str], ollama_model: str = None, skip_frames: bool = False) -> Dict[str, Any]:
        """
        Generate structured notes from chunks of specified sessions.
        
        Args:
            session_ids: List of session IDs to generate notes from
            ollama_model: Optional Ollama model to use (defaults to instance model)
            skip_frames: If True, skip frame data and use only transcript
            
        Returns:
            Dict with generated notes and metadata
        """
        result = {
            "success": False,
            "notes": None,
            "session_count": 0,
            "chunks_used": 0,
            "errors": []
        }
        
        if not HAS_OLLAMA:
            result["errors"].append("Ollama not installed. Run: pip install ollama")
            return result
        
        model = ollama_model or self.ollama_model
        
        # Get chunks for sessions
        chunks_result = self.get_chunks_for_sessions(session_ids)
        
        if not chunks_result["success"]:
            result["errors"] = chunks_result["errors"]
            return result
        
        if not chunks_result["chunks"]:
            result["errors"].append("No indexed content found for the selected sessions")
            return result
        
        try:
            # Build context from chunks (respect token limit)
            context_parts = []
            total_tokens = 0
            session_titles = {}
            
            # Check for VIDEO sessions and include frame data (unless skipped)
            frame_data_parts = []
            
            if not skip_frames:
                for session_id in session_ids:
                    # Check if this is a VIDEO session by looking for frame_data.txt
                    session_frame_data_file = MEDIA_VIDEO_DIR / session_id / "frame_data.txt"
                    
                    if session_frame_data_file.exists():
                        try:
                            with open(session_frame_data_file, 'r', encoding='utf-8') as f:
                                frame_data = f.read()
                                if frame_data.strip():
                                    frame_data_parts.append(f"=== Video Frame Analysis ===\n{frame_data}")
                                    print(f"  📼 Found frame data for session {session_id[:8]}...")
                        except Exception as e:
                            print(f"  ⚠️ Could not read frame data for {session_id}: {e}")
            else:
                print(f"  ⏭️ Skipping frame data as requested")
            
            # Add frame data to context if available
            if frame_data_parts:
                print(f"  📼 Including frame data from {len(frame_data_parts)} video session(s)")
                context_parts.append("\n\n=== VIDEO FRAME ANALYSIS ===\n\n" + "\n\n".join(frame_data_parts))
            elif skip_frames:
                print(f"  ⏭️ Skipping frame data, using transcript only")
            
            for chunk in chunks_result["chunks"]:
                chunk_tokens = self._estimate_tokens(chunk["text"])
                
                # Limit total context to prevent overwhelming the model
                # When including frame data, use fewer tokens for transcript
                max_tokens = 4000 if not frame_data_parts else 3000
                
                if total_tokens + chunk_tokens > max_tokens:
                    break
                
                context_parts.append(chunk["text"])
                total_tokens += chunk_tokens
                
                # Track session titles
                session_titles[chunk["session_id"]] = chunk["title"]
            
            context_string = "\n\n---\n\n".join(context_parts)
            result["chunks_used"] = len(context_parts)
            result["session_count"] = len(chunks_result["sessions_found"])
            
            # Build prompt for notes generation
            system_prompt = """You are an expert note-taker and study assistant. Given the following content from study sessions, 
create comprehensive, well-organized notes that:
1. Summarize the key concepts and main ideas
2. Use clear headings and bullet points
3. Highlight important definitions, formulas, or key terms
4. Group related information together
5. Be concise but complete

Format the notes in a clean, readable style suitable for studying."""

            session_list = ", ".join(session_titles.values()) if session_titles else "Selected sessions"
            user_prompt = f"""Create study notes from the following content (from: {session_list}):

{context_string}

Generate comprehensive study notes from this content."""

            # Call Ollama
            print(f"📝 Generating notes with Ollama ({model})...")
            
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            result["notes"] = response["message"]["content"]
            result["success"] = True
            print(f"  ✅ Generated notes ({len(result['notes'])} chars)")
            
        except Exception as e:
            error_msg = str(e)
            if "connection refused" in error_msg.lower() or "connect" in error_msg.lower():
                result["errors"].append("Ollama is not running. Start it with: ollama serve")
            else:
                result["errors"].append(f"Notes generation failed: {error_msg}")
            print(f"  ❌ Notes generation error: {e}")
        
        return result
    
    @staticmethod
    def get_capabilities() -> Dict[str, bool]:
        """Return available RAG capabilities."""
        return {
            "chromadb": HAS_CHROMADB,
            "sentence_transformers": HAS_SENTENCE_TRANSFORMERS,
            "ollama": HAS_OLLAMA,
            "fully_available": HAS_CHROMADB and HAS_SENTENCE_TRANSFORMERS and HAS_OLLAMA
        }


# Singleton instance
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine(ollama_model: str = "qwen3.5:2b") -> RAGEngine:
    """Get or create the singleton RAG engine instance."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine(ollama_model=ollama_model)
    return _rag_engine


if __name__ == "__main__":
    # Quick test
    print("RAG Engine Capabilities:", RAGEngine.get_capabilities())
    
    engine = get_rag_engine()
    
    # Test chunking
    test_text = """
    Machine learning is a subset of artificial intelligence. It enables computers to learn from data.
    
    Deep learning is a subset of machine learning. It uses neural networks with multiple layers.
    These networks can learn complex patterns from large amounts of data.
    
    Natural language processing (NLP) is another important field. It focuses on the interaction 
    between computers and human language. NLP enables applications like chatbots and translation.
    """
    
    chunks = engine.chunk_text(test_text)
    print(f"\nTest chunking: {len(chunks)} chunks created")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk)} chars, ~{engine._estimate_tokens(chunk)} tokens")
