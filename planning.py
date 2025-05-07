import json, logging
log = logging.getLogger(__name__)

def make_plan(llm, question, n, context, done, gaps):
    prompt = (
        f"QUESTION PRINCIPALE : {question}\n"
        f"CONTEXTE :\n{context or 'aucun'}\n"
        f"SOUS QUESTIONS DÉJÀ TRAITÉES : {', '.join(done) or 'aucune'}\n"
        f"ASPECTS MANQUANTS : {', '.join(gaps) or 'aucun'}\n\n"
        f"exemple : Propose EXACTEMENT {n} sous-questions JSON.\nExemple: [\"Sous-question 1\", \"Sous-question 2\"]"
    )
    out = llm.chat([{"role": "user", "content": prompt}])
    try:
        subs = json.loads(out)
    except Exception:
        subs = []
    subs = [s for s in subs if s and s not in done]

    # Si aucune nouvelle sous-question : on arrête proprement
    if not subs:
        return []
    return subs[:n]