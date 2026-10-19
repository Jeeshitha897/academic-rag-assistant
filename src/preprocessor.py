"""
Academic Document Preprocessing & Domain-Tailored Chunking Module
Project: Academic Doubt-Solving RAG System
Author: Lanka Jeeshita | Guide: Dr. P. Sudhakar
"""

import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


@dataclass
class AcademicChunk:
    chunk_id: str
    source: str
    page: int
    section: str
    text: str
    char_count: int
    token_estimate: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AcademicPreprocessor:
    """
    Domain-specific preprocessor designed for academic lecture notes,
    syllabi, and university textbooks.
    """

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Academic section regex patterns
        self.section_patterns = [
            re.compile(r"^(?:UNIT|CHAPTER|MODULE)\s+[0-9IVXLCDM]+[:\-\s].*", re.IGNORECASE),
            re.compile(r"^(?:SECTION|TOPIC)\s+[0-9]+(?:\.[0-9]+)*[:\-\s].*", re.IGNORECASE),
            re.compile(r"^(?:DEFINITION|THEOREM|ALGORITHM|LEMMA|COROLLARY|PROOF|EXAMPLE)\b.*", re.IGNORECASE),
            re.compile(r"^[0-9]+(?:\.[0-9]+)+\s+[A-Z].*"),
        ]

    def clean_text(self, text: str) -> str:
        """Cleans academic text while preserving formulas and punctuation."""
        if not text:
            return ""
        # Fix hyphenated words broken across line wraps (e.g., "trans-\nformer" -> "transformer")
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Normalize carriage returns and excessive empty lines
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Normalize multiple spaces (avoiding newline destruction)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def detect_section_header(self, line: str) -> Optional[str]:
        """Detects if a given line is an academic section or unit header."""
        cleaned = line.strip()
        if len(cleaned) < 3 or len(cleaned) > 120:
            return None
        for pattern in self.section_patterns:
            if pattern.match(cleaned):
                return cleaned
        return None

    def extract_from_pdf(self, file_path_or_buffer, source_name: str = "Uploaded_Document.pdf") -> List[Dict[str, Any]]:
        """
        Extracts text page-by-page from a PDF with metadata.
        """
        if PdfReader is None:
            raise ImportError("pypdf is required for PDF extraction. Please install with: pip install pypdf")

        reader = PdfReader(file_path_or_buffer)
        pages_data = []

        for page_idx, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            cleaned = self.clean_text(raw_text)
            if cleaned:
                pages_data.append({
                    "source": source_name,
                    "page": page_idx + 1,
                    "text": cleaned
                })

        return pages_data

    def extract_from_text(self, text_content: str, source_name: str = "Course_Notes.txt") -> List[Dict[str, Any]]:
        """
        Extracts sections from plain text or markdown files.
        """
        cleaned = self.clean_text(text_content)
        # Simulate pages if text is large (approx 2000 chars per page)
        page_char_limit = 2000
        pages_data = []

        if len(cleaned) <= page_char_limit:
            pages_data.append({
                "source": source_name,
                "page": 1,
                "text": cleaned
            })
        else:
            paragraphs = cleaned.split("\n\n")
            current_page_text = []
            current_len = 0
            page_num = 1

            for p in paragraphs:
                p_len = len(p)
                if current_len + p_len > page_char_limit and current_page_text:
                    pages_data.append({
                        "source": source_name,
                        "page": page_num,
                        "text": "\n\n".join(current_page_text)
                    })
                    page_num += 1
                    current_page_text = [p]
                    current_len = p_len
                else:
                    current_page_text.append(p)
                    current_len += p_len

            if current_page_text:
                pages_data.append({
                    "source": source_name,
                    "page": page_num,
                    "text": "\n\n".join(current_page_text)
                })

        return pages_data

    def chunk_academic_content(self, pages_data: List[Dict[str, Any]]) -> List[AcademicChunk]:
        """
        Splits academic document pages into semantic chunks respecting
        section boundaries, paragraph integrity, and sentence structure.
        """
        chunks: List[AcademicChunk] = []
        global_chunk_idx = 0
        current_section = "General Introduction"

        for page_item in pages_data:
            source = page_item["source"]
            page = page_item["page"]
            text = page_item["text"]

            lines = text.split("\n")
            current_buffer = []
            current_buffer_len = 0

            for line in lines:
                # Check for section headers to update current context
                detected = self.detect_section_header(line)
                if detected:
                    current_section = detected

                line_len = len(line)

                # If adding this line exceeds chunk_size, emit a chunk
                if current_buffer_len + line_len > self.chunk_size and current_buffer:
                    chunk_text = "\n".join(current_buffer).strip()
                    if len(chunk_text) >= 50:  # Ignore tiny noise chunks
                        global_chunk_idx += 1
                        chunks.append(AcademicChunk(
                            chunk_id=f"chunk_{global_chunk_idx:04d}",
                            source=source,
                            page=page,
                            section=current_section,
                            text=chunk_text,
                            char_count=len(chunk_text),
                            token_estimate=max(1, len(chunk_text) // 4)
                        ))

                    # Retain overlap for context continuity
                    overlap_buffer = []
                    overlap_len = 0
                    for prev_line in reversed(current_buffer):
                        if overlap_len + len(prev_line) <= self.chunk_overlap:
                            overlap_buffer.insert(0, prev_line)
                            overlap_len += len(prev_line)
                        else:
                            break
                    current_buffer = overlap_buffer
                    current_buffer_len = overlap_len

                current_buffer.append(line)
                current_buffer_len += line_len + 1

            # Emit remaining content in buffer
            if current_buffer:
                chunk_text = "\n".join(current_buffer).strip()
                if len(chunk_text) >= 40:
                    global_chunk_idx += 1
                    chunks.append(AcademicChunk(
                        chunk_id=f"chunk_{global_chunk_idx:04d}",
                        source=source,
                        page=page,
                        section=current_section,
                        text=chunk_text,
                        char_count=len(chunk_text),
                        token_estimate=max(1, len(chunk_text) // 4)
                    ))

        return chunks
