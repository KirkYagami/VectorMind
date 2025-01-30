import os
from pathlib import Path


def pull_ollama_embed_model(model_name: str = "nomic-embed-text"):
    model_path = Path.home() / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library"

    if not (model_path / model_name).exists():
        print(f"Model {model_name} not found. Pulling the model...")
        os.system(f"ollama pull {model_name}")
    else:
        print(f"Model {model_name} is already available locally.")
    return True

def force_pull_ollama_embed_model(model_name: str = "nomic-embed-text"):
    print(f"Force pulling model {model_name}...")
    os.system(f"ollama pull {model_name}")
    return True

from langchain_community.document_loaders import WikipediaLoader

def load_wikipedia_docs(query: str = "Transformer (deep learning architecture)",
                        max_docs: int = 2):
    
    docs = WikipediaLoader(query=query, load_max_docs=max_docs).load()
    return docs


urls = [
"https://towardsdatascience.com/transformers-141e32e69591",
"https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/",
"https://medium.com/@amanatulla1606/transformer-architecture-explained-2c49e2257b4c",
]
