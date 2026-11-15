"""Run every attack on every question and record the outcome."""
import time
import pandas as pd

from .rag import RAGTarget
from .documents import documents
from .static_attacks import INJECTIONS, make_poisoned
from .judge import judge

MODEL = "liquid/lfm-3-1.7b:free" # one specific OpenRouter model id, kept fixed
PAUSE = 3 # seconds between calls (free tier allows 20 per minute)

def run_one(question, attack, defense):
    """One question, one attack, one defense setting -> (outcome, answer)."""
    poisoned = attack != "clean"
    docs = list(documents)
    if poisoned:
        docs.append(make_poisoned(question, INJECTIONS[attack]))
    rag = RAGTarget(documents=docs, model=MODEL, defense=defense, top_k=3)
    try:
        result = rag.generate(question)
    except Exception as e:
        print("  error:", e)
        return "failed", "", MODEL
    time.sleep(PAUSE)
    return judge(result, poisoned=poisoned), result["answer"], result["model"]

def run_grid(questions, defenses=("none", "prompt", "filter", "both"), attacks=None):
    attacks = attacks or ["clean", *INJECTIONS]
    rows = []
    for defense in defenses:
        for attack in attacks:
            for item in questions:
                q = item["question"] if isinstance(item, dict) else item
                print(defense, attack, q[:40])
                outcome, answer, model = run_one(q, attack, defense)
                rows.append(dict(defense=defense, attack=attack, question=q,
                                 outcome=outcome, answer=answer, model=model))
    return pd.DataFrame(rows)

def run_doc(question, poison, defense):
    """Like run_one, but takes the poisoned document directly."""
    rag = RAGTarget(documents=list(documents) + [poison], model=MODEL, defense=defense, top_k=3)
    try:
        result = rag.generate(question)
    except Exception as e:
        print(f"  error ({type(e).__name__}): {e}")
        return "failed", ""
    time.sleep(PAUSE)
    return judge(result), result["answer"]