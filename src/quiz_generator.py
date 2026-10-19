"""
Academic Quiz & Revision Flashcards Generator
Project: Academic Doubt-Solving RAG System
Author: Lanka Jeeshita | Guide: Dr. P. Sudhakar
"""

import os
import json
import random
from typing import List, Dict, Any, Optional
from .preprocessor import AcademicChunk


class AcademicQuizGenerator:
    """
    Generates practice viva questions, MCQs, and revision flashcards
    directly from indexed lecture notes and textbook chapters.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model_name = model_name

    def set_api_key(self, api_key: str):
        self.api_key = api_key.strip()

    def generate_quiz(self, chunks: List[AcademicChunk], num_questions: int = 3) -> List[Dict[str, Any]]:
        """
        Generates MCQs with options, correct answer, and explanation.
        """
        if not chunks:
            return []

        # Select a representative sample of chunks
        sample_chunks = random.sample(chunks, min(len(chunks), 4))
        context_text = "\n\n".join([f"[{c.source} p.{c.page}]: {c.text}" for c in sample_chunks])

        if self.api_key:
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)
                prompt = f"""You are a university professor creating an examination quiz.
Based strictly on the following course material excerpts, generate {num_questions} multiple-choice questions (MCQs).

Course Material:
{context_text}

Return ONLY valid JSON matching this exact structure:
[
  {{
    "question": "Question text here?",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "correct_answer": "A",
    "explanation": "Detailed academic explanation grounded in the text.",
    "source_topic": "Section or concept name"
  }}
]
"""
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                text = response.text or ""
                # Strip markdown json backticks if present
                clean_text = text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]
                if clean_text.startswith("```"):
                    clean_text = clean_text[3:]
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]

                return json.loads(clean_text.strip())
            except Exception as e:
                print(f"[Warning] LLM Quiz generation failed: {e}. Falling back to template generation.")

        # Offline Fallback Generator
        quiz = []
        for i, c in enumerate(sample_chunks[:num_questions]):
            snippet = c.text[:120].strip()
            quiz.append({
                "question": f"Based on {c.section} (Page {c.page}), which statement best describes this concept?",
                "options": [
                    f"A) {snippet}...",
                    f"B) It represents an unrelated statistical outlier mechanism.",
                    f"C) It is deprecated in modern Generative AI pipelines.",
                    f"D) None of the above."
                ],
                "correct_answer": "A",
                "explanation": f"According to {c.source} page {c.page}, the correct explanation is explicitly stated in {c.section}.",
                "source_topic": c.section
            })
        return quiz

    def generate_flashcards(self, chunks: List[AcademicChunk], count: int = 4) -> List[Dict[str, str]]:
        """
        Generates quick revision flashcards (Concept -> Explanation).
        """
        if not chunks:
            return []

        selected = random.sample(chunks, min(len(chunks), count))
        flashcards = []
        for c in selected:
            # First line or section as term
            term = c.section if c.section != "General Introduction" else f"Key Concept ({c.source} p.{c.page})"
            summary = c.text[:250].strip() + "..."
            flashcards.append({
                "front": f"Explain: {term}",
                "back": f"{summary}\n\n[Reference: {c.source}, Page {c.page}]"
            })
        return flashcards
