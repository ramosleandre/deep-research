import json, logging
log = logging.getLogger(__name__)

SYSTEM_PLANNER = (
    "Tu es un planificateur. Décompose la QUESTION en {n} sous-questions JSON."
)

def make_plan(llm, question: str, n: int = 3) -> list[str]:
    prompt = SYSTEM_PLANNER.format(n=n) + "\nQUESTION: " + question
    out = llm.chat([{"role": "user", "content": prompt}])
    try:
        subs = json.loads(out)
    except Exception:
        subs = [question]
    log.info("Sous-questions: %s", subs)
    return subs[:n]