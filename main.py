import argparse, logging
from providers.openai_provider import OpenAIProvider
from providers.ollama_provider import OllamaProvider
from deep_research import DeepResearcher

def get_llm(provider: str, model: str | None):
    if provider == "openai":
        return OpenAIProvider(model or "gpt-4o-mini")
    return OllamaProvider(model or "llama3:8b")

def cli():
    ap = argparse.ArgumentParser("deep_research")
    ap.add_argument("-q", "--question", required=True)
    ap.add_argument("-s", "--site", help="URL racine à explorer (sinon web)")    
    ap.add_argument("-p", "--provider", choices=["openai", "ollama"], default="openai")
    ap.add_argument("--model", help="Nom du modèle (OpenAI ou Ollama)")
    ap.add_argument("-m", "--max_tasks", type=int, default=3)
    ap.add_argument("--max_iters", type=int, default=2,
                    help="Nombre d’itérations plan→collecte→replan (≥1)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    llm = get_llm(args.provider, args.model)
    dr = DeepResearcher(
        llm, args.question, args.site,
        max_tasks=args.max_tasks, max_iters=args.max_iters
    )
    answer = dr.run()
    print("\n=== RÉPONSE FINALE ===\n")
    print(answer)

if __name__ == "__main__":
    cli()