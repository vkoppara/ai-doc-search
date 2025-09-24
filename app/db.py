import os
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")  

def get_connection():
    return psycopg2.connect(DB_URL)

def create_tables():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                chunk TEXT NOT NULL,
                vector_embedding vector,
                chunk_index INTEGER NOT NULL
            );
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                code TEXT PRIMARY KEY,
                message TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                email_id TEXT NOT NULL
            );
            """)
        conn.commit()
    print("Tables created successfully.")