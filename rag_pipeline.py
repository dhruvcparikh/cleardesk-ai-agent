import re
# --- CHUNKING --- #

def chunk_text(text, separator="\n\n"):
    return [s.strip() for s in text.split(separator) if len(s.strip()) > 50]

import os
all_chunks = []
for filename in os.listdir("files"):
    with open(f"files/{filename}", "r") as f:
        text = f.read()
    chunks = chunk_text(text)
    for chunk in chunks:
        all_chunks.append({"text": chunk, "source": filename})

print(f"Total chunks: {len(all_chunks)}")
print(f"Example chunk: {all_chunks[1]['text'][:500]}...")


# --- EMBEDDING --- #

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# # Show what a vector looks like:
# sample = get_embedding("How do I reset my password in ClearDesk?")
# print(f"Vector length: {len(sample)}")    
# print(f"First 10 values: {sample[:10]}")  


# Then embed all chunks and save:
import json
for chunk in all_chunks:
    chunk["embedding"] = get_embedding(chunk["text"])
with open("embeddings.json", "w") as f:
    json.dump(all_chunks, f)
print(f"Saved {len(all_chunks)} embeddings to embeddings.json")


# --- RETRIEVING --- #

import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, top_k=3):
    query_embedding = get_embedding(query)
    scored = []
    for chunk in all_chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

# Test it:
results = retrieve("How do I export my project data as CSV?")
for score, chunk in results:
    print(f"Score: {score:.4f} | Source: {chunk['source']}")
    print(f"Text:  {chunk['text'][:150]}...")
    print()
# Expected: top chunk from cleardesk_faq.txt, score ~0.89


#  --- GENERATION --- #

def ask(question):
    results = retrieve(question, top_k=3)
    context = "\n\n---\n\n".join([chunk["text"] for _, chunk in results])
    sources = [chunk["source"] for _, chunk in results]

    response = client.chat.completions.create(
        model="gpt-4o", temperature=0,
        messages=[
            {"role": "system", "content": f"""You are a customer support agent
for ClearDesk. Answer ONLY from the provided documentation.
If the answer isn't there, say so and offer to escalate.
Documentation:
{context}"""},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content, sources

# Run TWO full tests:
q1 = "How do I export my project data as CSV?"
answer, sources = ask(q1)
print(f"Q: {q1}\nA: {answer}\nSources: {sources}")

# q2 = "Why am I not able to add another project?"
# answer, sources = ask(q2)
# print(f"\nQ: {q2}\nA: {answer}\nSources: {sources}")
