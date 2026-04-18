"""
PDFRAGTool: Smolagents tool for RAG-based PDF document querying
Allows agents to search and retrieve information from PDF documents
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import numpy as np

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import smolagents
smolagents_path = parent_dir / "smolagents" / "src"
sys.path.insert(0, str(smolagents_path))

from smolagents import tool

# Import core components
from core_src.embedding_generator import EmbeddingGenerator

try:
    import PyPDF2
    import faiss
except ImportError:
    print("Warning: PyPDF2 and/or faiss not installed. Install with: pip install PyPDF2 faiss-cpu")


class PDFRAGSystem:
    """RAG system for PDF documents"""

    def __init__(self, base_path: str = "./data/pdf_documents",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        """
        Initialize PDF RAG system

        Args:
            base_path: Base path for PDF document storage (default: ./data/pdf_documents)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.base_path = Path(base_path)
        self.pdfs_path = self.base_path / "pdfs"
        self.metadata_path = self.base_path / "metadata"
        self.index_path = self.base_path / "rag_index"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Create directory structure
        self.pdfs_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(
            model_name="PE-Core-L14-336",
            device="cuda",
            normalize=True
        )

        # Load or create FAISS index
        self.index = None
        self.chunks = []
        self.metadata = []

        self._load_or_create_index()

    def _load_or_create_index(self):
        """Load existing index or create new one"""
        try:
            index_file = self.index_path / "faiss_index.bin"
            metadata_file = self.index_path / "metadata.json"

            if index_file.exists() and metadata_file.exists():
                # Load existing index
                self.index = faiss.read_index(str(index_file))

                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    self.chunks = data["chunks"]
                    self.metadata = data["metadata"]

                print(f"Loaded existing index with {len(self.chunks)} chunks")
            else:
                # Create new index (1024-dimensional for PE-Core)
                self.index = faiss.IndexFlatIP(1024)  # Inner product for normalized vectors
                self.chunks = []
                self.metadata = []
                print("Created new FAISS index")

        except Exception as e:
            print(f"Error loading/creating index: {e}")
            self.index = faiss.IndexFlatIP(1024)
            self.chunks = []
            self.metadata = []

    def add_pdf(self, pdf_path: str) -> int:
        """
        Add PDF to the RAG system

        Args:
            pdf_path: Path to PDF file

        Returns:
            Number of chunks added
        """
        try:
            import shutil
            from datetime import datetime

            print(f"Processing PDF: {pdf_path}")

            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                print(f"PDF file not found: {pdf_path}")
                return 0

            pdf_name = pdf_path.name

            # Copy PDF to storage if not already there
            stored_pdf_path = self.pdfs_path / pdf_name
            if not stored_pdf_path.exists():
                shutil.copy2(pdf_path, stored_pdf_path)
                print(f"Copied PDF to: {stored_pdf_path}")
            else:
                print(f"PDF already exists in storage: {stored_pdf_path}")

            # Extract text from PDF
            text = self._extract_text_from_pdf(str(stored_pdf_path))

            if not text.strip():
                print(f"Warning: No text extracted from {pdf_name}")
                return 0

            # Chunk text
            chunks = self._chunk_text(text)

            if not chunks:
                print(f"Warning: No chunks generated from {pdf_name}")
                return 0

            # Generate embeddings
            embeddings = []
            for chunk in chunks:
                embedding = self.embedding_generator.encode_text(chunk)
                embeddings.append(embedding)

            embeddings = np.array(embeddings)

            # Add to FAISS index
            self.index.add(embeddings)

            # Store chunks and metadata
            chunk_start_id = len(self.chunks)
            for i, chunk in enumerate(chunks):
                self.chunks.append(chunk)
                self.metadata.append({
                    "source": pdf_name,
                    "chunk_id": len(self.chunks) - 1,
                    "chunk_index": i
                })

            # Create PDF metadata file
            pdf_metadata_path = self.metadata_path / f"{pdf_path.stem}.json"
            pdf_metadata = {
                "filename": pdf_name,
                "original_path": str(pdf_path),
                "stored_path": str(stored_pdf_path),
                "file_size": stored_pdf_path.stat().st_size,
                "added_at": datetime.now().isoformat(),
                "num_chunks": len(chunks),
                "chunk_ids": list(range(chunk_start_id, chunk_start_id + len(chunks))),
                "total_characters": len(text),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }

            with open(pdf_metadata_path, 'w') as f:
                json.dump(pdf_metadata, f, indent=2)

            # Save index
            self._save_index()

            print(f"Added {len(chunks)} chunks from {pdf_name}")
            print(f"Metadata saved to: {pdf_metadata_path}")
            return len(chunks)

        except Exception as e:
            print(f"Error adding PDF: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        try:
            text = ""
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        try:
            chunks = []
            words = text.split()

            current_chunk = []
            current_length = 0

            for word in words:
                current_chunk.append(word)
                current_length += len(word) + 1

                if current_length >= self.chunk_size:
                    chunks.append(" ".join(current_chunk))
                    # Keep overlap
                    overlap_words = int(len(current_chunk) * (self.chunk_overlap / self.chunk_size))
                    current_chunk = current_chunk[-overlap_words:] if overlap_words > 0 else []
                    current_length = sum(len(w) + 1 for w in current_chunk)

            # Add last chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            return chunks

        except Exception as e:
            print(f"Error chunking text: {e}")
            return []

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of results with text and metadata
        """
        try:
            if self.index.ntotal == 0:
                return []

            # Generate query embedding
            query_embedding = self.embedding_generator.encode_text(query)
            query_embedding = query_embedding.reshape(1, -1)

            # Search
            scores, indices = self.index.search(query_embedding, top_k)

            # Prepare results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks):
                    results.append({
                        "text": self.chunks[idx],
                        "score": float(score),
                        "metadata": self.metadata[idx]
                    })

            return results

        except Exception as e:
            print(f"Error searching: {e}")
            return []

    def _save_index(self):
        """Save FAISS index and metadata"""
        try:
            index_file = self.index_path / "faiss_index.bin"
            metadata_file = self.index_path / "metadata.json"

            faiss.write_index(self.index, str(index_file))

            with open(metadata_file, 'w') as f:
                json.dump({
                    "chunks": self.chunks,
                    "metadata": self.metadata
                }, f)

        except Exception as e:
            print(f"Error saving index: {e}")

    def list_pdfs(self) -> List[Dict[str, Any]]:
        """
        List all PDFs in the system

        Returns:
            List of PDF metadata dictionaries
        """
        pdfs = []

        try:
            # Scan metadata directory
            for metadata_file in self.metadata_path.glob("*.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    pdfs.append(metadata)
                except Exception as e:
                    print(f"Error loading metadata from {metadata_file}: {e}")

            # Sort by added_at (newest first)
            pdfs.sort(key=lambda x: x.get("added_at", ""), reverse=True)

        except Exception as e:
            print(f"Error listing PDFs: {e}")

        return pdfs

    def get_pdf_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific PDF

        Args:
            filename: PDF filename (with or without extension)

        Returns:
            PDF metadata dictionary or None
        """
        try:
            # Remove extension if present and add .json
            stem = Path(filename).stem
            metadata_file = self.metadata_path / f"{stem}.json"

            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            else:
                print(f"Metadata file not found: {metadata_file}")
                return None

        except Exception as e:
            print(f"Error getting PDF metadata: {e}")
            return None

    def remove_pdf(self, filename: str) -> bool:
        """
        Remove a PDF from the system

        Args:
            filename: PDF filename to remove

        Returns:
            True if successful, False otherwise
        """
        try:
            stem = Path(filename).stem

            # Get metadata to find chunk IDs
            metadata = self.get_pdf_metadata(filename)
            if not metadata:
                print(f"Cannot remove PDF: metadata not found for {filename}")
                return False

            # Note: Removing from FAISS requires rebuilding the index
            # For now, we'll just mark it as removed and rebuild if needed
            print(f"Warning: Removing PDF from FAISS index requires rebuilding")
            print(f"PDF chunks will remain in index until rebuild")

            # Remove from global metadata list
            chunk_ids = set(metadata.get("chunk_ids", []))
            self.metadata = [m for m in self.metadata if m.get("chunk_id") not in chunk_ids]
            self.chunks = [c for i, c in enumerate(self.chunks) if i not in chunk_ids]

            # Save updated index
            self._save_index()

            # Remove files
            pdf_file = self.pdfs_path / filename
            if pdf_file.exists():
                pdf_file.unlink()
                print(f"Removed PDF file: {pdf_file}")

            metadata_file = self.metadata_path / f"{stem}.json"
            if metadata_file.exists():
                metadata_file.unlink()
                print(f"Removed metadata file: {metadata_file}")

            return True

        except Exception as e:
            print(f"Error removing PDF: {e}")
            import traceback
            traceback.print_exc()
            return False


# Global RAG system instance
_rag_system = None
_selected_pdf_files = []  # List of selected PDF filenames


def get_rag_system() -> PDFRAGSystem:
    """Get or create global RAG system instance"""
    global _rag_system
    if _rag_system is None:
        _rag_system = PDFRAGSystem()
    return _rag_system


def set_selected_pdf_files(pdf_files: list):
    """
    Set the selected PDF files for searches

    Args:
        pdf_files: List of PDF filenames to search in
    """
    global _selected_pdf_files
    _selected_pdf_files = pdf_files if pdf_files else []


def get_selected_pdf_files() -> list:
    """
    Get the selected PDF files

    Returns:
        List of selected PDF filenames
    """
    return _selected_pdf_files


def enforce_selected_pdf_contexts(pdf_sources: Optional[list] = None) -> Dict[str, Any]:
    """
    Enforce that only selected PDF contexts can be queried.

    Args:
        pdf_sources: Optional list of PDF filenames to validate.
                    If None, will use selected PDFs.
                    If provided, validates they're in selected PDFs.

    Returns:
        Dict with "status", "pdf_sources", "message"
        - If status == "error": Operation should stop
        - If status == "success": Use the returned pdf_sources list
    """
    selected_pdfs = get_selected_pdf_files()

    # If no PDFs selected at all
    if not selected_pdfs:
        return {
            "status": "error",
            "pdf_sources": [],
            "message": "No PDFs selected. Please select PDFs from Context Catalog tab first. Use get_selected_contexts() to check."
        }

    # If pdf_sources explicitly provided, validate they're in selected
    if pdf_sources is not None:
        # Check if any requested PDFs are not in selected
        not_selected = [pdf for pdf in pdf_sources if pdf not in selected_pdfs]

        if not_selected:
            return {
                "status": "error",
                "pdf_sources": [],
                "message": f"Cannot query PDFs {not_selected} - they are not selected. Selected PDFs: {selected_pdfs}. Use get_selected_contexts() to see selected contexts."
            }

        # All provided PDFs are valid
        return {
            "status": "success",
            "pdf_sources": pdf_sources,
            "message": f"Using {len(pdf_sources)} selected PDF(s)"
        }

    # Use all selected PDFs (pdf_sources was None)
    return {
        "status": "success",
        "pdf_sources": selected_pdfs,
        "message": f"Using {len(selected_pdfs)} selected PDF(s)"
    }


@tool
def pdf_rag_search(query: str, pdf_sources: Optional[list] = None, top_k: int = 5) -> str:
    """
    Search PDF documents using RAG (Retrieval Augmented Generation).
    Finds relevant information from uploaded PDF documents based on the query.
    Can filter search to specific PDF files.

    Args:
        query: The search query text
        pdf_sources: Optional list of PDF filenames to search in. If not provided, uses selected PDFs.
                    If empty list provided, searches all PDFs.
        top_k: Number of top results to return (default: 5)

    Returns:
        A JSON string containing relevant context from PDF documents
    """
    try:
        rag_system = get_rag_system()

        # Enforce selected contexts
        context_check = enforce_selected_pdf_contexts(pdf_sources)
        if context_check["status"] == "error":
            return json.dumps({
                "status": "error",
                "message": context_check["message"]
            })

        # Use validated PDF sources from selected contexts
        pdf_sources = context_check["pdf_sources"]

        # Search (get more results initially for filtering)
        search_k = top_k * 3 if pdf_sources else top_k
        results = rag_system.search(query, search_k)

        if not results:
            return json.dumps({
                "status": "no_results",
                "message": "No relevant information found in PDF documents",
                "pdf_sources": pdf_sources,
                "results": []
            })

        # Filter by selected PDFs (always filter since we enforced selection)
        results = [r for r in results if r["metadata"]["source"] in pdf_sources]

        # Take top_k after filtering
        results = results[:top_k]

        if not results:
            return json.dumps({
                "status": "no_results",
                "message": f"No relevant information found in selected PDFs: {pdf_sources}",
                "pdf_sources": pdf_sources,
                "results": []
            })

        # Format results
        formatted_results = []
        sources_used = set()
        for i, result in enumerate(results):
            source = result["metadata"]["source"]
            sources_used.add(source)
            formatted_results.append({
                "rank": i + 1,
                "text": result["text"],
                "relevance_score": result["score"],
                "source": source
            })

        return json.dumps({
            "status": "success",
            "query": query,
            "pdf_sources": pdf_sources if pdf_sources else "all",
            "sources_used": list(sources_used),
            "num_results": len(formatted_results),
            "results": formatted_results
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Error searching PDFs: {str(e)}"
        })


@tool
def add_pdf_to_rag(pdf_path: str) -> str:
    """
    Add a PDF document to the RAG system for future querying.

    Args:
        pdf_path: Path to the PDF file to add

    Returns:
        A JSON string with the status of the operation
    """
    try:
        rag_system = get_rag_system()

        # Add PDF
        num_chunks = rag_system.add_pdf(pdf_path)

        if num_chunks > 0:
            return json.dumps({
                "status": "success",
                "message": f"Successfully added PDF to RAG system",
                "pdf_path": pdf_path,
                "num_chunks": num_chunks
            }, indent=2)
        else:
            return json.dumps({
                "status": "error",
                "message": "Failed to add PDF to RAG system"
            })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Error adding PDF: {str(e)}"
        })
