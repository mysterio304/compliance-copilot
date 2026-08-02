from dataclasses import dataclass
from functools import lru_cache

from FlagEmbedding import BGEM3FlagModel


@dataclass(frozen=True)
class EmbeddingResult:
    dense: list[float]
    sparse: dict[int, float]


@lru_cache(maxsize=1)
def _get_model(model_name: str) -> BGEM3FlagModel:
    """Cached per model name: loading BGE-M3 takes real time, so a process should
    only ever pay it once."""
    return BGEM3FlagModel(model_name, use_fp16=False)


def embed_texts(texts: list[str], model_name: str) -> list[EmbeddingResult]:
    """Embeds with BGE-M3 in one pass: dense (for cosine similarity) and sparse
    lexical weights (token_id -> weight, for lexical/exact-term matching) — the
    same model doing both is the reason BGE-M3 was chosen over a dense-only model.
    """
    model = _get_model(model_name)
    output = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
    results = []
    for dense, sparse in zip(output["dense_vecs"], output["lexical_weights"], strict=True):
        sparse_int = {
            int(token_id): float(weight) for token_id, weight in sparse.items() if weight > 0
        }
        results.append(EmbeddingResult(dense=dense.tolist(), sparse=sparse_int))
    return results


def embed_query(text: str, model_name: str) -> EmbeddingResult:
    return embed_texts([text], model_name)[0]
