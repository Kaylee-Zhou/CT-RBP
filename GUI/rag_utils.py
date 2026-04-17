import os
import logging
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

logger = logging.getLogger(__name__)

# Initialize encoder
encoder = SentenceTransformer("./all-MiniLM-L6-v2")

# Load knowledge base
KNOWLEDGE_FILE = "knowledge_base.txt"
if os.path.exists(KNOWLEDGE_FILE):
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
    logger.info(f"Knowledge base loaded. Total {len(docs)} entries.")
else:
    logger.warning(f"Knowledge base file {KNOWLEDGE_FILE} not found. Using empty knowledge base.")
    docs = []

# Build FAISS index
if docs:
    doc_embeddings = encoder.encode(docs, convert_to_numpy=True)
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)
else:
    index = None

# API configuration (hardcoded as original)
API_KEY = "sk-2ca600b2a0be428a81dd5c94f6517d8b"
API_BASE = "https://api.deepseek.com"
API_MODEL = "deepseek-chat"
client = OpenAI(api_key=API_KEY, base_url=API_BASE)


def rag_explain(sequence, predicted_label, predicted_prob, top_k=2):
    # Search knowledge base
    retrieved_docs = []
    if docs and index:
        try:
            query_text = f"RNA sequence: {sequence}"
            query_emb = encoder.encode([query_text], convert_to_numpy=True)
            scores, indices = index.search(query_emb, top_k)
            if indices.shape[1] > 0:
                for i in indices[0]:
                    if 0 <= i < len(docs):
                        retrieved_docs.append(docs[i])
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            retrieved_docs = []
    else:
        retrieved_docs = ["Knowledge base is empty, no experimental methods retrieved."]

    seq_short = str(sequence)[:50]
    pred_text = "binding" if predicted_label == 1 else "non-binding"

    knowledge_text = "\n\n".join([f"Method {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])

    prompt = f"""You are an expert RNA biologist. Analyze this prediction and suggest validation experiments.

## Prediction Input
- RNA sequence (k-mer perspective): {seq_short}
- Model prediction: {pred_text} (probability: {predicted_prob:.4f})

## Retrieved Experimental Methods (from knowledge base)
{knowledge_text}

## Your Task
Based on the RNA sequence features and the retrieved experimental methods above, provide:

1. Mechanistic Explanation: Why does this RNA likely {pred_text}? Consider sequence motifs, structural features, or biological context. (2-3 sentences)

2. Wet-lab Validation Plan: Propose 2 specific experiments to validate this prediction. For each:
   - Which method to use (choose from retrieved methods above, or combine them)
   - Brief procedure outline
   - Expected outcome if the prediction is correct

Keep your response concise (under 200 words) and technically accurate."""

    try:
        response = client.chat.completions.create(
            model=API_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert RNA biologist specializing in protein-RNA interactions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300,
            timeout=60  # Increased timeout to avoid fetch failure
        )
        explanation = response.choices[0].message.content.strip()

        parsed_info = {
            'sequence_preview': seq_short,
            'predicted': f"{pred_text} ({predicted_prob:.4f})"
        }

        return {
            'status': 'success',
            'explanation': explanation,
            'retrieved_knowledge': retrieved_docs,
            'parsed_info': parsed_info
        }
    except Exception as e:
        logger.error(f"LLM call failed: {e}", exc_info=True)
        return {
            'status': 'failed',
            'error': f"Language model generation failed: {str(e)}",
            'retrieved_knowledge': retrieved_docs
        }