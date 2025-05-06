from typing import List, Dict
import ollama
import logging, time

log = logging.getLogger(__name__)

class OllamaProvider:
    def __init__(self, model: str = "llama3:8b"):
        self.model = model
        self._ensure_model()

    def _ensure_model(self):
        try:
            ollama.show(self.model)
        except:
            log.warning("Pull Ollama model %s …", self.model)
            ollama.pull(self.model)

    def chat(self, messages: List[Dict[str, str]], temperature: float = .2) -> str:
        t0 = time.time()
        resp = ollama.chat(model=self.model, messages=messages, options={"temperature": temperature})
        log.debug("Ollama %.1fs", time.time() - t0)
        return resp["message"]["content"]