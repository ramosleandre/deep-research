import json, logging
log = logging.getLogger(__name__)

# SYSTEM_PLANNER = (
#     "Tu es un planificateur de recherche.\n"
#     "CONTEXTE_COUPLÉ:\n{context}\n"
#     "Sous‑questions déjà traitées: {done}\n"
#     "Propose maintenant {n} nouvelles sous‑questions JSON, sans répéter le déjà‑couvert sur des choses que tu ne métrises pas par rapport au contexte."
# )

def make_plan(llm, question, n, context, done, gaps):
    prompt = (
        "Tu es l’agent planificateur.\n"
        f"CONTEXTE COUVERT :\n{context or 'aucun'}\n"
        f"SOUS QUESTIONS DÉJÀ TRAITÉES : {', '.join(done) or 'aucune'}\n"
        f"ASPECTS MANQUANTS IMPORTANTS : {', '.join(gaps) or 'aucun'}\n\n"
        f"Propose EXACTEMENT {n} nouvelles sous questions JSON, "
        "chacune ciblant un aspect manquant.\n"
        "Exemple : [\"Sous question 1\", \"Sous question 2\"]"
    )
    out = llm.chat([{"role": "user", "content": prompt}])
    try:
        subs = json.loads(out)
    except Exception:
        subs = []
    # filtre doublons
    subs = [s for s in subs if s and s not in done]
    # fallback : si vide, renvoie la question brute
    return subs[:n] or [question]