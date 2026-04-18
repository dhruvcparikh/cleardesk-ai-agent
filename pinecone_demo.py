import os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


# --- CHUNKING --- #

def chunk_text(text, separator="\n\n"):
    return [s.strip() for s in text.split(separator) if len(s.strip()) > 50]

all_chunks = []
for filename in os.listdir("files"):
    with open(f"files/{filename}", "r") as f:
        text = f.read()
    chunks = chunk_text(text)
    for chunk in chunks:
        all_chunks.append({"text": chunk, "source": filename})

# print(f"Total chunks: {len(all_chunks)}")
# print(f"Example chunk: {all_chunks[1]['text'][:500]}...")


# --- EMBEDDING --- #

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


# --- PINECONE SETUP --- #
# After first run, comment out create_index + upsert to avoid re-uploading

# pc.create_index(
#     name="cleardesk-knowledge",
#     dimension=1536,
#     metric="cosine",
#     spec=ServerlessSpec(cloud="aws", region="us-east-1")
# )

index = pc.Index("cleardesk-knowledge")

# vectors = []
# for i, chunk in enumerate(all_chunks):
#     vectors.append({
#         "id": f"chunk_{i}",
#         "values": get_embedding(chunk["text"]),
#         "metadata": {"text": chunk["text"], "source": chunk["source"]}
#     })
# index.upsert(vectors=vectors, batch_size=100)
# print(f"Uploaded {len(vectors)} vectors to Pinecone")


# --- RETRIEVING --- #

def retrieve(query, top_k=3):
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [(m.score, m.metadata) for m in results.matches]


# ============================================================
# MEMORY APPROACH SWITCH
# Options: "full_history" | "sliding_window" | "summary"
# ============================================================
MEMORY_APPROACH = "summary"
WINDOW_SIZE = 3      # only used when MEMORY_APPROACH = "sliding_window"
SUMMARY_KEEP = 4     # recent turns to keep verbatim in summary mode


# --- CONVERSATION STORE --- #

conversations = {}

def start_conversation():
    conv_id = str(len(conversations) + 1)
    conversations[conv_id] = []
    return conv_id


# --- APPROACH 1: FULL HISTORY --- #

def ask_with_memory(conv_id, question):
    results = retrieve(question, top_k=3)
    context = "\n\n---\n\n".join([meta["text"] for _, meta in results])
    messages = [{"role": "system", "content": f"ClearDesk support. Answer ONLY from docs.\nDocs: {context}"}]
    for turn in conversations[conv_id]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})
    response = client.chat.completions.create(model="gpt-4o", temperature=0, messages=messages)
    answer = response.choices[0].message.content
    conversations[conv_id].append({"role": "user", "content": question})
    conversations[conv_id].append({"role": "assistant", "content": answer})
    return answer


# --- APPROACH 2: SLIDING WINDOW --- #

def ask_with_window(conv_id, question):
    results = retrieve(question, top_k=3)
    context = "\n\n---\n\n".join([meta["text"] for _, meta in results])
    messages = [{"role": "system", "content": f"ClearDesk support. Answer ONLY from docs.\nDocs: {context}"}]
    for turn in conversations[conv_id][-WINDOW_SIZE:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})
    response = client.chat.completions.create(model="gpt-4o", temperature=0, messages=messages)
    answer = response.choices[0].message.content
    conversations[conv_id].append({"role": "user", "content": question})
    conversations[conv_id].append({"role": "assistant", "content": answer})
    return answer


# --- APPROACH 3: SUMMARY MEMORY --- #

def summarize_history(history):
    if len(history) < SUMMARY_KEEP + 2:
        return None
    old_turns = history[:-SUMMARY_KEEP]
    old_text = "\n".join([f"{t['role']}: {t['content']}" for t in old_turns])
    response = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        messages=[
            {"role": "system", "content": "Summarize this support conversation in 2-3 sentences. Preserve: core issue, account details (plan tier, workspace count), troubleshooting steps tried."},
            {"role": "user", "content": old_text}
        ]
    )
    return response.choices[0].message.content

def ask_with_summary(conv_id, question):
    results = retrieve(question, top_k=3)
    context = "\n\n---\n\n".join([meta["text"] for _, meta in results])
    summary = summarize_history(conversations[conv_id])
    system = f"""ClearDesk support. Answer ONLY from docs.
Respond in JSON with these fields:
- category: the ticket category : product, process, pricing, billing, user management
- sentiment: positive, neutral, or negative
- response: your answer to the customer
- escalate: true or false
- confidence: a number from 0 to 1
\nDocs: {context}"""
    if summary:
        system += f"\n\nConversation context: {summary}"
    messages = [{"role": "system", "content": system}]
    for turn in conversations[conv_id][-SUMMARY_KEEP:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})
    response = client.chat.completions.create(model="gpt-4o", temperature=0, messages=messages)
    answer = response.choices[0].message.content
    conversations[conv_id].append({"role": "user", "content": question})
    conversations[conv_id].append({"role": "assistant", "content": answer})
    return answer


# --- UNIFIED ASK FUNCTION --- #

def ask(conv_id, question):
    print(f"[Mode: {MEMORY_APPROACH}] Q: {question}")
    if MEMORY_APPROACH == "full_history":
        answer = ask_with_memory(conv_id, question)
    elif MEMORY_APPROACH == "sliding_window":
        answer = ask_with_window(conv_id, question)
    elif MEMORY_APPROACH == "summary":
        answer = ask_with_summary(conv_id, question)
    else:
        raise ValueError(f"Unknown MEMORY_APPROACH: '{MEMORY_APPROACH}'. Choose: full_history | sliding_window | summary")
    print(f"A: {answer}\n")
    return answer


# --- RUN DEMO --- #

# --- RUN DEMO --- #

conv = start_conversation()

# Turn 1-2: Establish key account context (this is what gets "forgotten" in sliding window)
ask(conv, "Hi, I'm on the Enterprise plan with 14 workspaces and 200 users.")
ask(conv, "Our account ID is ENT-8821 and we're based in the EU region.")

# Turn 3-4: Start a support issue
ask(conv, "I'm trying to export project data as CSV for all 14 workspaces at once.")
ask(conv, "I followed the docs but got a 403 error on the bulk export endpoint.")

# Turn 5-6: Go deeper into troubleshooting (window starts dropping turns 1-2 here)
ask(conv, "I'm a workspace admin, so it shouldn't be a permissions issue.")
ask(conv, "I also tried from two different browsers and got the same 403.")

# Turn 7: This is where sliding window fails — "my plan" and "14 workspaces" are gone
# Summary mode still knows because it summarized turns 1-4
print("--- Turn 7: sliding window should forget Enterprise plan + workspace count ---")
ask(conv, "Does my plan allow 14 workspaces? I'm not sure")
