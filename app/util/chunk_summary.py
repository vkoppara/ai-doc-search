# embedding_summary.py

import numpy as np
from sklearn.cluster import KMeans
from .openai_client import get_chat_content


def extractive_summary(chunks, embeddings, num_summary_chunks=5):
    """
    Generate an extractive summary from chunk embeddings using KMeans clustering.
    """
    if len(chunks) == 0 or len(embeddings) == 0:
        return ""

    # Ensure embeddings is a 2D NumPy array
    embeddings = np.array(embeddings)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(len(chunks), -1)
    
    num_summary_chunks = min(num_summary_chunks, len(chunks))
    
    # KMeans clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_summary_chunks, random_state=42)
    kmeans.fit(embeddings)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Function to pick the chunk closest to cluster centroid
    def closest_chunk_to_centroid(cluster_idx):
        cluster_chunks_idx = np.where(labels == cluster_idx)[0]
        if len(cluster_chunks_idx) == 0:
            return None
        cluster_embeddings = embeddings[cluster_chunks_idx]
        distances = np.linalg.norm(cluster_embeddings - cluster_centers[cluster_idx], axis=1)
        closest_idx_in_cluster = cluster_chunks_idx[np.argmin(distances)]
        return closest_idx_in_cluster

    summary_indices = []
    for i in range(num_summary_chunks):
        idx = closest_chunk_to_centroid(i)
        if idx is not None:
            summary_indices.append(idx)

    summary_chunks = [chunks[i] for i in summary_indices]
    return summary("\n\n".join(summary_chunks))


def summary(summary_chunks):
    system_prompt = (
        """
        You are an AI assistant specialized in generating concise and accurate executive summaries for documents. 

        Your task is to create a **document-level summary** using the provided **chunks of text**, which together represent the full content of the document. 

        Requirements:

        1. The summary should **capture the main purpose, key points, and overall content** of the document.  
        2. The chunks are **parts of the same document**, but may not be in perfect order; use them collectively to understand the document.  
        3. Do **not include extraneous information** that is not present in the chunks.  
        4. Keep the summary **concise, clear, and professional**, suitable for an executive or manager to quickly understand what the document is about.  
        5. Preserve **important entities, names, or numbers** mentioned in the chunks, if relevant.  

        Input: A list of text chunks representing the document.  
        Output: A concise executive summary of the document, written in natural language.
        """
    )
    try:
        answer = get_chat_content(
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": f"chunks of documnet: {summary_chunks}"}],
            model="sonar", max_retries=3, temperature=0.7, max_tokens=4096
        )
        return answer
    except Exception as e:
         return f"Exception Occurred, {e}"




# Example usage
if __name__ == "__main__":
    # Example chunks and embeddings
    chunks = ["Chunk 1 text ...", "Chunk 2 text ...", "Chunk 3 text ..."]
    embeddings = np.array([
        [0.1, 0.3, 0.5],
        [0.2, 0.1, 0.4],
        [0.05, 0.25, 0.6]
    ])
    
    summary = extractive_summary(chunks, embeddings, num_summary_chunks=2)
    print("=== Extractive Summary ===")
    print(summary)
