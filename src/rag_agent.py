"""
Academic RAG Agent & LLM Query Handler
Project: Academic Doubt-Solving RAG System
Author: Lanka Jeeshita | Guide: Dr. P. Sudhakar
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple


ACADEMIC_SYSTEM_PROMPT = """You are the "Academic Doubt-Solving AI Assistant", an intelligent pedagogical teaching assistant designed for university students.
Your primary objective is to resolve academic doubts strictly based on the provided course lecture notes, syllabus, and textbooks.

Guidelines for Academic Responses:
1. Grounding & Zero-Hallucination: Answer the student's question ONLY using the factual context provided in the 'Course Material Excerpts' below. Do not fabricate facts.
2. Source Attribution: Explicitly cite your references throughout your explanation using the exact format: `[Doc: <document_name>, Page: <page_number>]`.
3. Pedagogical Tone: Structure your answers like an exemplary professor:
   - Clear High-Level Definition / Concept Summary
   - Step-by-Step Mathematical Formulation or Architectural Breakdown (use LaTeX for formulas)
   - Real-World or Practical Example
   - Key Takeaway / Exam Tip
4. Out-of-Syllabus Handling: If the query cannot be answered from the provided excerpts, state clearly:
   "⚠️ Note: This specific topic is not found in the uploaded course material. Based on your syllabus, here are the closest related topics..."
   Do not guess.
"""


class AcademicRAGAgent:
    """
    RAG Agent managing intent detection, context assembly,
    prompting, and response generation with source verification.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model_name = model_name

    def set_api_key(self, api_key: str):
        self.api_key = api_key.strip()

    def classify_intent(self, query: str) -> str:
        """
        Classifies the student's query to customize explanation style.
        """
        q_lower = query.lower()
        if any(w in q_lower for w in ["derive", "derivation", "formula", "equation", "proof", "mathematical"]):
            return "FORMULA_DERIVATION"
        elif any(w in q_lower for w in ["code", "python", "implementation", "syntax", "algorithm", "function"]):
            return "CODE_IMPLEMENTATION"
        elif any(w in q_lower for w in ["exam", "revision", "summary", "important questions", "key points"]):
            return "EXAM_REVISION"
        elif any(w in q_lower for w in ["who are you", "hello", "hi", "hey", "help"]):
            return "GREETING"
        else:
            return "CONCEPTUAL_EXPLANATION"

    def _build_context_prompt(self, query: str, retrieved_items: List[Dict[str, Any]], tone: str = "Detailed") -> str:
        """
        Formats retrieved course excerpts with page numbers and sections.
        """
        context_parts = []
        for i, item in enumerate(retrieved_items, 1):
            chunk = item["chunk"]
            context_parts.append(
                f"--- EXCERPT {i} ---\n"
                f"Document: {chunk.source} | Page: {chunk.page} | Section: {chunk.section}\n"
                f"Content:\n{chunk.text}\n"
            )

        context_str = "\n".join(context_parts)

        tone_instruction = {
            "Concise Revision": "Provide a quick, concise revision summary suitable for quick review before class.",
            "Detailed": "Provide an in-depth academic explanation with definitions, step-by-step points, and examples.",
            "Exam-Oriented": "Structure the answer in standard university exam format (Definition, Key Characteristics, Diagram Description, and Summary)."
        }.get(tone, "Provide an in-depth academic explanation.")

        prompt = f"""{ACADEMIC_SYSTEM_PROMPT}

Student Query Tone Preference: {tone_instruction}

Course Material Excerpts:
{context_str}

Student Question:
{query}

Please formulate an academic response citing the sources with `[Doc: ..., Page: ...]`:
"""
        return prompt

    def generate_offline_extractive_answer(
        self, query: str, retrieved_items: List[Dict[str, Any]], intent: str
    ) -> Dict[str, Any]:
        """
        Intelligent extractive synthesizer for 100% offline or zero-API-key demonstrations.
        Extracts key sentences matching the query from the top chunks with citations.
        """
        if not retrieved_items:
            return {
                "answer": "No relevant course material found. Please upload lecture notes or textbooks in the Course Library tab.",
                "sources": [],
                "confidence": 0.0,
                "intent": intent,
                "mode": "Offline Extractive Engine"
            }

        sources = []
        extracted_points = []
        q_words = set(re.findall(r"\b\w{3,}\b", query.lower()))

        for item in retrieved_items:
            chunk = item["chunk"]
            sources.append({
                "source": chunk.source,
                "page": chunk.page,
                "section": chunk.section,
                "score": round(item["score"], 4),
                "snippet": chunk.text[:200] + "..."
            })

            # Score sentences within this chunk
            sentences = re.split(r"(?<=[.?!])\s+", chunk.text)
            for s in sentences:
                s_words = set(re.findall(r"\b\w{3,}\b", s.lower()))
                overlap = len(q_words.intersection(s_words))
                if overlap > 0:
                    clean_s = s.strip()
                    if clean_s and clean_s not in [p[1] for p in extracted_points]:
                        extracted_points.append((overlap, clean_s, chunk.source, chunk.page, chunk.section))

        extracted_points.sort(key=lambda x: x[0], reverse=True)
        top_sentences = extracted_points[:5]

        # Compose academic response
        lead_chunk = retrieved_items[0]["chunk"]
        answer_lines = [
            f"### 📘 Academic Concept Summary",
            f"According to **{lead_chunk.source}** (Page {lead_chunk.page}, *{lead_chunk.section}*):\n",
        ]

        if top_sentences:
            for _, s, doc, pg, sec in top_sentences:
                answer_lines.append(f"- {s} `[Doc: {doc}, Page: {pg}]`")
        else:
            # Fallback to direct excerpt
            answer_lines.append(f"{lead_chunk.text}\n\n`[Doc: {lead_chunk.source}, Page: {lead_chunk.page}]`")

        answer_lines.append("\n\n> 💡 *Note: Running in local extractive mode. Enter a Gemini API Key in the sidebar for full conversational LLM synthesis.*")

        confidence = min(0.95, max(0.40, retrieved_items[0]["score"] * 30))

        return {
            "answer": "\n".join(answer_lines),
            "sources": sources,
            "confidence": round(confidence, 2),
            "intent": intent,
            "mode": "Offline Extractive Engine"
        }

    def answer_query(
        self,
        query: str,
        retrieved_items: List[Dict[str, Any]],
        tone: str = "Detailed"
    ) -> Dict[str, Any]:
        """
        Executes end-to-end RAG answer generation.
        """
        intent = self.classify_intent(query)

        if intent == "GREETING":
            return {
                "answer": "Hello! I am your **Academic Doubt-Solving Assistant** for Generative AI & Coursework.\n\nAsk me any concept, formula derivation, or syllabus doubt from your lecture notes and textbooks!",
                "sources": [],
                "confidence": 1.0,
                "intent": intent,
                "mode": "Interactive"
            }

        if not retrieved_items:
            return {
                "answer": "⚠️ No relevant excerpts could be retrieved from your course documents. Please check that you have uploaded lecture notes in the Document Manager tab.",
                "sources": [],
                "confidence": 0.0,
                "intent": intent,
                "mode": "Not Found"
            }

        # Check if API key is provided
        if not self.api_key:
            return self.generate_offline_extractive_answer(query, retrieved_items, intent)

        # Full Generative LLM synthesis via Google GenAI SDK
        prompt = self._build_context_prompt(query, retrieved_items, tone)

        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            raw_text = response.text or "I could not generate an answer from the retrieved notes."

            sources = []
            for item in retrieved_items:
                chunk = item["chunk"]
                sources.append({
                    "source": chunk.source,
                    "page": chunk.page,
                    "section": chunk.section,
                    "score": round(item["score"], 4),
                    "snippet": chunk.text[:220] + "..."
                })

            top_score = retrieved_items[0]["score"] if retrieved_items else 0.0
            confidence = min(0.98, max(0.60, top_score * 35))

            return {
                "answer": raw_text,
                "sources": sources,
                "confidence": round(confidence, 2),
                "intent": intent,
                "mode": f"Live LLM ({self.model_name})"
            }

        except Exception as e:
            print(f"[Error] Gemini API generation error: {e}. Falling back to extractive mode.")
            offline_res = self.generate_offline_extractive_answer(query, retrieved_items, intent)
            offline_res["answer"] = f"⚠️ *API call encountered an issue ({e}). Switched to local offline extraction:*\n\n" + offline_res["answer"]
            return offline_res
