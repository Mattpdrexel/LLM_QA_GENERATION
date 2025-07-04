import pandas as pd
import numpy as np
import random
import re
import json
import ollama
import os
from typing import List, Dict, Tuple, Any

# ──────────────── CONFIG ──────────────── #

EXCEL_PATH     = "data/raw_data/salem_cw_data.xlsx"
OUTPUT_PATH    = "data/questions_and_answers/qapairs.jsonl"
QUESTION_MODEL = "mistral-small3.1:latest" #"llama2:13b"
ANSWER_MODEL   = "mistral-small3.1:latest" #"llama2:13b"
EMBED_MODEL    = "nomic-embed-text:latest"      # for embeddings
EMBEDDINGS_CACHE_PATH = "data/embeddings.npy"
NUM_ROUNDS     = 50
SAMPLE_SIZE    = 1
TOP_K          = 50

# ──────────────── LOAD CW TRAINING MATERIAL ──────────────── #
CW_TRAINING_PATH = "data/raw_data/cw_training.txt"
with open(CW_TRAINING_PATH, "r", encoding="utf-8") as f:
    CW_TRAINING = f.read()

# ──────────────── QUESTION PROMPT TEMPLATE ──────────────── #

# ──────────────── PROMPT TEMPLATES ──────────────── #
QUESTION_GEN_PROMPT_TEMPLATE = """
# ─── Background ─────────────────────────────────────────
{background}

# ─── Examples ───────────────────────────────────────────
Example 1:
Q: {seed1_question}
A: {seed1_answer}

Example 2:
Q: {seed2_question}
A: {seed2_answer}

# ─── Context Notifications ─────────────────────────────
{component_context}

# ─── Your Task ─────────────────────────────────────────
Using the notifications above and focusing on component "{component}", craft one single-sentence question that is both insightful and grounded. Return only the question.
"""

QA_GEN_PROMPT_TEMPLATE = """
# ─── Background ─────────────────────────────────────────
{background}

# ─── Examples ───────────────────────────────────────────
Example 1:
Q: {seed1_question}
A: {seed1_answer}

Example 2:
Q: {seed2_question}
A: {seed2_answer}

# ─── Context Notifications ─────────────────────────────
{question_context}

# ─── Original Question ───────────────────────────────────
{question}

# ─── Your Task ─────────────────────────────────────────
Based on the context notifications and the original question above, generate a new single-sentence question that is insightful and grounded, and provide a concise answer.  
Return a JSON object with keys "question" and "answer".  
You MUST include a reference section with the context notifications that led to the answer in the format: Not. Notification – ShortText (CreatedOn).
We will use these Q&A pairs to fine-tune a later model, so please return strictly valid JSON.
"""

# ──────────────── COMBINED QA PROMPT TEMPLATE ──────────────── #
COMBINED_QA_PROMPT_TEMPLATE = """
# ─── Background ─────────────────────────────────────────
{background}

# ─── Examples ───────────────────────────────────────────
Example 1:
Q: {seed1_question}
A: {seed1_answer}

Example 2:
Q: {seed2_question}
A: {seed2_answer}

# ─── Component Context Notifications ────────────────────
{component_context}

# ─── Your Task ─────────────────────────────────────────
Focusing on component "{component}", craft one single-sentence question that is both insightful and grounded based on the above notifications, then provide a concise answer.
Return a JSON object with keys "question", "answer", and "references" where "references" is the list of notifications cited (e.g., "Not. 123 – ShortText (YYYY-MM-DD)").
We will use these Q&A pairs to fine-tune a later model, so please return strictly valid JSON.
"""

# ──────────────── UTILITIES ──────────────── #

def load_excel_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    # normalize Notification to str
    df["Notification"] = df["Notification"].astype(int).astype(str)
    return df

def build_embeddings(df: pd.DataFrame) -> np.ndarray:
    # load cached embeddings if available
    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        return np.load(EMBEDDINGS_CACHE_PATH)
    """Embed each ShortText+LongText via Ollama."""
    # combine key columns for richer embedding context
    cols_to_embed = ["Notification", "CreatedOn", "OrderNum", "FLOC", "ShortText", "LongText"]
    # ensure datetime to string for CreatedOn
    df_embed = df.copy()
    if "CreatedOn" in df_embed.columns:
        df_embed["CreatedOn"] = df_embed["CreatedOn"].astype(str)
    texts = (
        df_embed[cols_to_embed]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .tolist()
    )
    embs = []
    for t in texts:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=t)
        embs.append(resp["embedding"])
    arr = np.array(embs)
    # save embeddings for future runs
    np.save(EMBEDDINGS_CACHE_PATH, arr)
    return arr

def cosine_sim_matrix(embs: np.ndarray, query_emb: np.ndarray) -> np.ndarray:
    dot = embs @ query_emb
    norm = np.linalg.norm(embs, axis=1) * np.linalg.norm(query_emb)
    return dot / norm

def extract_primary_component(snippet: str) -> str:
    """Use the LLM to pick one primary component from the snippet."""
    prompt = f"""
Here are several notification entries:

{snippet}

Identify the single primary component these notifications relate to.
Your answer must:
- Include a specific equipment identifier and descriptor if appropriate (e.g., "11A CWP", "12A Travel Screen", "13B Screenwash Pump").
- Not be a generic term like "pump", "lube oil", "motor", etc.

Return only the exact component string.
"""
    resp = ollama.generate(model=QUESTION_MODEL, prompt=prompt)["response"].strip()
    # attempt to isolate the component string
    comp = resp
    # if there's a colon, take text after the last colon
    if ":" in comp:
        comp = comp.split(":")[-1]
    # remove markdown formatting like asterisks
    comp = comp.replace("*", "")
    # strip quotes and whitespace
    return comp.strip(' "\'')

# ────────── STEP 1: SAMPLE & QUESTION GENERATION ────────── #

# load hand-curated seed Q&A examples
SEEDS_PATH = "data/questions_and_answers/seeds.json"
with open(SEEDS_PATH, "r", encoding="utf-8") as f:
    SEEDS = json.load(f)

def retrieve_relevant_semantic(df: pd.DataFrame, embeddings: np.ndarray,
                               query: str, top_k=TOP_K) -> pd.DataFrame:
    qemb = np.array(ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"])
    sims = cosine_sim_matrix(embeddings, qemb)
    idx_top = np.argsort(sims)[-top_k:][::-1]
    return df.iloc[idx_top]

def get_context_snippet(df: pd.DataFrame) -> str:
    return "\n".join(
        f"Not. {r.Notification} – {r.ShortText} ({r.CreatedOn.date()})"
        for _, r in df.iterrows()
    )

def pipeline_round(df: pd.DataFrame, embeddings: np.ndarray) -> Dict:
    # LLM call #1: extract component
    sample = df.sample(1).iloc[0]
    snippet = " | ".join(
        f"{col}: {sample[col]}" for col in ["Notification","CreatedOn","OrderNum","FLOC","ShortText","LongText"] if col in sample
    )
    primary_comp = extract_primary_component(snippet)

    # retrieve context for QA
    comp_context_df = retrieve_relevant_semantic(df, embeddings, primary_comp, top_k=TOP_K)
    component_context = get_context_snippet(comp_context_df)

    # LLM call #2: generate combined Q&A
    resp = ollama.generate(
        model=ANSWER_MODEL,
        prompt=COMBINED_QA_PROMPT_TEMPLATE.format(
            background=CW_TRAINING,
            seed1_question=SEEDS[0]["question"],
            seed1_answer=SEEDS[0]["answer"],
            seed2_question=SEEDS[1]["question"],
            seed2_answer=SEEDS[1]["answer"],
            component=primary_comp,
            component_context=component_context
        )
    )["response"].strip()

    try:
        qa_json = json.loads(resp)
    except Exception:
        # fallback regex
        q_match = re.search(r'"question"\s*:\s*"(.+?)"', resp)
        a_match = re.search(r'"answer"\s*:\s*"(.+?)"', resp)
        refs = re.findall(r'Not\. \d+ – [^\n]+', resp)
        qa_json = {
            "question": q_match.group(1) if q_match else None,
            "answer": a_match.group(1) if a_match else None,
            "references": refs
        }

    # ensure references field exists
    if "references" not in qa_json:
        qa_json["references"] = component_context.split("\n")

    return qa_json

# ────────── MAIN ────────── #

def main():
    df         = load_excel_data(EXCEL_PATH)
    embeddings = build_embeddings(df)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for _ in range(NUM_ROUNDS):
            qa = pipeline_round(df, embeddings)
            fout.write(json.dumps(qa) + "\n")
            print("✔", qa["question"])

    print(f"\nGenerated {NUM_ROUNDS} Q&A pairs → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
