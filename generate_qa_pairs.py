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
QUESTION_MODEL = "mistral-small3.1:latest"      # or your preferred model
ANSWER_MODEL   = "mistral-small3.1:latest"      # or your preferred model
EMBED_MODEL    = "nomic-embed-text:latest"      # for embeddings
EMBEDDINGS_CACHE_PATH = "data/embeddings.npy"
NUM_ROUNDS     = 1000
SAMPLE_SIZE    = 5
TOP_K          = 50

# ──────────────── OPEN ISSUE FILTER & TOP_K BY TYPE ──────────────── #
# mapping of question types to number of context items
QUESTION_TYPE_TOPK: Dict[str, int] = {
    "trend": TOP_K,
    "drilldown": TOP_K,
    "recent": TOP_K,
    "compare": TOP_K,
    "similar": TOP_K,
}

# ──────────────── QUESTION PROMPT TEMPLATE ──────────────── #
QUESTION_PROMPT_TEMPLATE = """
# ─── Examples ──────────────────────────────────────────────────
Example 1:
Q: {seed1_question}

Example 2:
Q: {seed2_question}

# ─── Sample Notifications ─────────────────────────────────────
+{snippet}

Primary component to focus on: **{primary_comp}**

# ─── Your Task ────────────────────────────────────────────────
For {primary_comp}, craft *one* single-sentence question that is both **insightful** and **grounded** in the details above.  
Your question should:
1. Start exactly with "For {primary_comp},"  
2. Reference at least one *specific detail* or anomaly from the snippet (e.g. a failure symptom, trend, or unusual reading)  
3. Invite analysis or comparison (e.g. "What could explain…", "How does X compare to Y…", "Which factors likely contribute to…")  
4. Be phrased as a single question ending with a question mark  
5. *Not* mention any dates, times, or other components  

# ─── Generate ────────────────────────────────────────────────
Return **only** the question text.
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

def filter_open_issues(df: pd.DataFrame) -> pd.DataFrame:
    # infer open issues: no work order and no resolution text
    cond = pd.Series(True, index=df.index)
    if "OrderNum" in df.columns:
        cond &= df["OrderNum"].isna() | (df["OrderNum"].astype(str) == "")
    # infer resolution from presence of keywords in ShortText or LongText
    keywords = ["repair", "replace", "fixed", "restored", "completed", "repaired"]
    pattern = "|".join(keywords)
    has_res = df["ShortText"].str.contains(pattern, case=False, na=False) \
             | df["LongText"].str.contains(pattern, case=False, na=False)
    cond &= ~has_res
    return df[cond]

def extract_primary_component(snippet: str) -> str:
    """Use the LLM to pick one primary component from the snippet."""
    prompt = f"""
Here are several notification entries:

{snippet}

Identify the single primary component these notifications relate to.
Your answer must:
- Include a specific equipment identifier and descriptor (e.g., "11A CWP", "12A Travel Screen", "13B Screenwash Pump").
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

def sample_and_generate_question(df: pd.DataFrame, num_samples=SAMPLE_SIZE) -> Tuple[str, str, str]:
    # choose among more targeted question styles
    question_type = random.choice([
        "trend",        # high-level patterns
        "drilldown",    # examples of one issue
        "recent",       # most recent issues for a component
        "compare",      # compare two components or issues
        "similar",      # find similar events by symptoms
    ])

    # select sample based on question type: only filter for open issues on 'recent'
    if question_type == "recent":
        base_df = filter_open_issues(df)
    else:
        base_df = df
    sample = base_df.sample(min(num_samples, len(base_df)))

    # build snippet reflecting full row values
    snippet = "\n".join(
        " | ".join(
            f"{col}: {r[col]}" for col in ["Notification", "CreatedOn", "OrderNum", "FLOC", "ShortText", "LongText"] if col in r
        )
        for _, r in sample.iterrows()
    )

    # determine a single primary component from the snippet
    primary_comp = extract_primary_component(snippet)
    print(f"Primary component: {primary_comp}\n")
    # build the question prompt using the unified template
    prompt = QUESTION_PROMPT_TEMPLATE.format(
        seed1_question=SEEDS[0]['question'],
        seed2_question=SEEDS[1]['question'],
        snippet=snippet,
        primary_comp=primary_comp
    )
    resp = ollama.generate(model=QUESTION_MODEL, prompt=prompt)
    return question_type, resp["response"].strip(), primary_comp

# ────────── STEP 2: SEMANTIC RETRIEVAL ────────── #

def retrieve_relevant_semantic(df: pd.DataFrame, embeddings: np.ndarray,
                               question: str, top_k=TOP_K) -> pd.DataFrame:
    # embed the question
    qemb = np.array(ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"])
    sims = cosine_sim_matrix(embeddings, qemb)
    idx_top = np.argsort(sims)[-top_k:][::-1]
    return df.iloc[idx_top]

# ────────── STEP 3: ANSWER GENERATION ────────── #

def format_refs(rows: pd.DataFrame) -> str:
    seen = set()
    lines: List[str] = []
    for _, r in rows.iterrows():
        nid = r.Notification
        if nid in seen:
            continue
        seen.add(nid)
        lines.append(f"Not. {nid} – {r.ShortText} ({r.CreatedOn.date()})")
    return "\n".join(lines)

def answer_question(question: str, relevant: pd.DataFrame) -> str:
    refs = format_refs(relevant)
    # few-shot seed examples for answer structure
    seed_ans = "\n\n".join(
        f"Example {i+1} Answer:\n{s['answer']}" for i, s in enumerate(SEEDS)
    )
    prompt = f"""
{seed_ans}

Using ONLY the following notifications as evidence:

{refs}

Question: {question}

First, provide a concise narrative answer addressing the question. Then list the supporting notifications in bullet points, citing each as Not. [Notification] – [ShortText] (YYYY-MM-DD). Return ONLY your answer.
"""
    # generate answer without unsupported parameters
    resp = ollama.generate(
        model=ANSWER_MODEL,
        prompt=prompt
    )
    return resp["response"].strip()

# ────────── STEP 4: REFERENCE VERIFICATION ────────── #

def verify_references(answer: str, df: pd.DataFrame) -> bool:
    ids = re.findall(r"Not\.\s*(\d+)", answer)
    valid = set(df["Notification"].tolist())
    return all(i in valid for i in ids)

# ────────── STEP 3.5: QUESTION REFINEMENT ────────── #
def refine_question(question: str, context: pd.DataFrame, primary_comp: str) -> str:
    # build reference snippet from context
    refs = format_refs(context)
    # enforce strict component mention and format in refinement and preserve intent
    prompt = f"""
Initial draft question: {question}

Using ONLY the following notifications as context:

{refs}

Rewrite the question to meet these criteria:
1) It must mention no other component
2) It must be a single sentence ending with a question mark, or a request for data (i.e., get me instances of event X related to component Y)
3) It must focus strictly on {primary_comp}
4) It must NOT mention any dates or time periods
5) It must preserve the analytical intent of the original draft question (e.g., comparing, identifying causes, finding similarities)
6) Do NOT simply ask for notifications, but rather ask for a specific event or symptom related to {primary_comp}
Return ONLY the improved question.
"""
    return ollama.generate(model=QUESTION_MODEL, prompt=prompt)["response"].strip()

def extract_components_from_question(question: str) -> List[str]:
    """Use LLM to identify component IDs referenced in a question and return as a list."""
    prompt = f"""
Identify the specific component identifier(s) referenced in the following question (e.g., '11A', '13B'). Return a JSON array of component names only.

Question: {question}
"""
    resp = ollama.generate(model=QUESTION_MODEL, prompt=prompt)["response"].strip()
    try:
        comps = json.loads(resp)
        if isinstance(comps, list):
            return [str(c) for c in comps]
    except Exception:
        pass
    return []

# ────────── FULL PIPELINE ROUND ────────── #

def pipeline_round(df: pd.DataFrame, embeddings: np.ndarray) -> Dict:
    # 1) Draft a question and get its type + primary component
    draft_question, question_type, primary_comp = sample_and_generate_question(df)
    # 2) Determine context DF and embeddings: filter open issues only for 'recent'
    if question_type == "recent":
        context_df = filter_open_issues(df)
        context_embs = embeddings[context_df.index.to_list()]
    else:
        context_df = df
        context_embs = embeddings
    # 3) Initial context retrieval with type-specific top_k
    k = QUESTION_TYPE_TOPK.get(question_type, TOP_K)
    initial_context = retrieve_relevant_semantic(context_df, context_embs, draft_question, top_k=k)
    # 4) Refine the question based on initial context and primary component
    question = refine_question(draft_question, initial_context, primary_comp)
    # 5) Filter context to rows that mention primary_comp in ShortText or LongText
    # escape regex special chars in primary component
    escaped_comp = re.escape(primary_comp)
    mask = (
        context_df['ShortText'].astype(str).str.contains(fr"\b{escaped_comp}\b", case=False, na=False)
        | context_df['LongText'].astype(str).str.contains(fr"\b{escaped_comp}\b", case=False, na=False)
    )
    filtered_df = context_df[mask]
    if not filtered_df.empty:
        idxs = filtered_df.index.to_list()
        filtered_embs = context_embs[idxs]
        relevant = retrieve_relevant_semantic(filtered_df, filtered_embs, question, top_k=k)
    else:
        relevant = retrieve_relevant_semantic(context_df, context_embs, question, top_k=k)

    # 6) Generate the answer
    answer = answer_question(question, relevant)

    # if any bad refs, retry once (verify against context_df)
    if not verify_references(answer, context_df):
        answer = answer_question(question, relevant)

    return {"question": question, "answer": answer}

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
