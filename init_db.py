"""
Database Initialization Script for Academic Doubt-Solving RAG
Indexes default course notes on startup.
"""

import os
from src.preprocessor import AcademicPreprocessor
from src.vector_engine import AcademicVectorEngine

def initialize_database():
    print("[INFO] Initializing Academic Vector Store...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_notes_path = os.path.join(base_dir, "sample_notes", "GenAI_Agentic_Systems_Course_Notes.txt")
    vector_db_dir = os.path.join(base_dir, "vector_db")

    if not os.path.exists(sample_notes_path):
        print(f"[ERROR] Sample notes not found at {sample_notes_path}")
        return

    with open(sample_notes_path, "r", encoding="utf-8") as f:
        content = f.read()

    preprocessor = AcademicPreprocessor(chunk_size=650, chunk_overlap=120)
    pages = preprocessor.extract_from_text(content, source_name="GenAI_Agentic_Systems_Notes.txt")
    chunks = preprocessor.chunk_academic_content(pages)
    print(f"[INFO] Generated {len(chunks)} academic chunks.")

    engine = AcademicVectorEngine()
    engine.build_index(chunks)
    engine.save_to_disk(vector_db_dir)
    print(f"[SUCCESS] Vector store saved successfully to {vector_db_dir} with {len(chunks)} chunks.")

if __name__ == "__main__":
    initialize_database()
