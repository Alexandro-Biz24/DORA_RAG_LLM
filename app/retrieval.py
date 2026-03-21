import re
import pickle
from typing import List, Dict, Optional

import faiss
import numpy as np
import pandas as pd

from app.config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    BM25_PATH,
    TOP_K_VECTOR,
    TOP_K_BM25,
    TOP_K_FINAL,
    USE_BM25,
)
from app.openai_client import client


# =========================================================
# LOADERS
# =========================================================

_faiss_index = None
_metadata_df = None
_bm25 = None


def load_faiss_index():
    global _faiss_index
    if _faiss_index is None:
        _faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
    return _faiss_index


def load_metadata():
    global _metadata_df
    if _metadata_df is None:
        _metadata_df = pd.read_parquet(METADATA_PATH)
    return _metadata_df


def load_bm25():
    global _bm25
    if _bm25 is None and BM25_PATH.exists():
        with open(BM25_PATH, "rb") as f:
            _bm25 = pickle.load(f)
    return _bm25


# =========================================================
# EMBEDDINGS
# =========================================================

def embed_query(query: str, model: str = EMBEDDING_MODEL) -> np.ndarray:
    response = client.embeddings.create(
        model=model,
        input=query
    )
    vec = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec


# =========================================================
# QUERY UNDERSTANDING / FILTER EXTRACTION
# =========================================================

def extract_query_filters(query: str) -> Dict[str, Optional[str]]:
    q = query.strip()

    article_match = re.search(r"\barticle\s+([0-9]+[a-zA-Z]?)\b", q, flags=re.IGNORECASE)
    chapter_match = re.search(r"\bchapter\s+([IVXLCDM0-9]+)\b", q, flags=re.IGNORECASE)
    section_match = re.search(r"\bsection\s+([IVXLCDM0-9]+)\b", q, flags=re.IGNORECASE)
    title_match = re.search(r"\btitle\s+([IVXLCDM0-9]+)\b", q, flags=re.IGNORECASE)

    # dates simples
    date_match = re.search(
        r"\b(\d{1,2}\s+[A-Za-zéûôàèîùç]+\s+\d{4}|\d{4})\b",
        q,
        flags=re.IGNORECASE
    )

    return {
        "article_nb": article_match.group(1) if article_match else None,
        "chapter_nb": chapter_match.group(1) if chapter_match else None,
        "section_nb": section_match.group(1) if section_match else None,
        "title_nb": title_match.group(1) if title_match else None,
        "date_hint": date_match.group(1) if date_match else None,
    }

def metadata_match_score(row: pd.Series, filters: Dict[str, Optional[str]], query: str) -> float:
    score = 0.0
    q_lower = query.lower()

    # hard boosts numéro
    if filters.get("article_nb") and str(row.get("article_nb")) == str(filters["article_nb"]):
        score += 5.0

    if filters.get("chapter_nb") and str(row.get("chapter_nb")) == str(filters["chapter_nb"]):
        score += 3.0

    if filters.get("section_nb") and str(row.get("section_nb")) == str(filters["section_nb"]):
        score += 2.5

    if filters.get("title_nb") and str(row.get("title_nb")) == str(filters["title_nb"]):
        score += 2.5

    # date / année
    if filters.get("date_hint"):
        date_hint = filters["date_hint"].lower()
        doc_date = str(row.get("doc_date") or "").lower()
        if date_hint in doc_date:
            score += 2.0
        else:
            # si juste l'année est contenue
            year_match = re.search(r"\b(\d{4})\b", date_hint)
            if year_match and year_match.group(1) in doc_date:
                score += 1.2

    # matching lexical sur metadata texte
    metadata_fields = [
        ("article_name", 2.0),
        ("section_name", 1.5),
        ("chapter_name", 1.5),
        ("title_name", 1.5),
        ("doc_title", 2.0),
        ("doc_subtitle", 1.8),
    ]

    query_tokens = [tok for tok in re.findall(r"\w+", q_lower) if len(tok) > 3]

    for col, boost in metadata_fields:
        val = row.get(col)
        if isinstance(val, str) and val:
            val_lower = val.lower()
            token_hits = sum(tok in val_lower for tok in query_tokens)
            score += min(token_hits * 0.35, boost)

    # léger boost article
    if row.get("block_type") == "article":
        score += 0.5

    return score
    
# =========================================================
# METADATA RE-SCORING
# =========================================================


def expand_with_full_blocks(results: List[Dict]) -> List[Dict]:
    metadata = load_metadata()

    block_keys = set()
    for item in results:
        block_keys.add((item["doc_id"], item["block_id"]))

    expanded = metadata[
        metadata.apply(lambda row: (row["doc_id"], row["block_id"]) in block_keys, axis=1)
    ].to_dict(orient="records")

    return expanded

# =========================================================
# VECTOR SEARCH
# =========================================================

def vector_search(query: str, top_k: int = TOP_K_VECTOR) -> List[Dict]:
    index = load_faiss_index()
    metadata = load_metadata()

    query_vec = embed_query(query)
    search_k = max(top_k * 3, 20)

    scores, indices = index.search(query_vec, search_k)

    results = []
    filters = extract_query_filters(query)

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0:
            continue

        row = metadata.iloc[idx].to_dict()

        meta_boost = metadata_match_score(pd.Series(row), filters, query)
        final_score = float(score) + meta_boost

        row["vector_score"] = float(score)
        row["metadata_boost"] = float(meta_boost)
        row["final_score"] = float(final_score)
        row["retrieval_source"] = "vector"
        row["faiss_idx"] = int(idx)
        row["rank"] = rank

        results.append(row)

    results = sorted(results, key=lambda x: x["final_score"], reverse=True)
    return results[:top_k]


# =========================================================
# BM25 SEARCH (optional)
# =========================================================

def bm25_search(query: str, top_k: int = TOP_K_BM25) -> List[Dict]:
    if not USE_BM25:
        return []

    bm25 = load_bm25()
    if bm25 is None:
        return []

    metadata = load_metadata()
    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)
    top_idx = np.argsort(scores)[::-1][:max(top_k * 3, 20)]

    filters = extract_query_filters(query)
    results = []

    for rank, idx in enumerate(top_idx):
        row = metadata.iloc[idx].to_dict()
        lexical_score = float(scores[idx])
        meta_boost = metadata_match_score(pd.Series(row), filters, query)
        final_score = lexical_score + meta_boost

        row["bm25_score"] = lexical_score
        row["metadata_boost"] = meta_boost
        row["final_score"] = final_score
        row["retrieval_source"] = "bm25"
        row["faiss_idx"] = int(idx)
        row["rank"] = rank

        results.append(row)

    results = sorted(results, key=lambda x: x["final_score"], reverse=True)
    return results[:top_k]


# =========================================================
# MERGE / DEDUP
# =========================================================

def merge_results(vector_results: List[Dict], bm25_results: List[Dict], top_k_final: int = TOP_K_FINAL) -> List[Dict]:
    merged = {}

    for item in vector_results + bm25_results:
        chunk_id = item["chunk_id"]

        if chunk_id not in merged:
            merged[chunk_id] = item
        else:
            # garde le meilleur score et fusionne les sources
            if item["final_score"] > merged[chunk_id]["final_score"]:
                prev_source = merged[chunk_id].get("retrieval_source", "")
                new_source = item.get("retrieval_source", "")
                item["retrieval_source"] = f"{prev_source}+{new_source}"
                merged[chunk_id] = item
            else:
                merged[chunk_id]["retrieval_source"] = (
                    f'{merged[chunk_id].get("retrieval_source", "")}+{item.get("retrieval_source", "")}'
                )

    merged_list = list(merged.values())
    merged_list = sorted(merged_list, key=lambda x: x["final_score"], reverse=True)

    return merged_list[:top_k_final]


# =========================================================
# CONTEXT EXPANSION
# =========================================================

def expand_with_neighbors(results: List[Dict]) -> List[Dict]:
    metadata = load_metadata()
    chunk_id_to_row = {row["chunk_id"]: row for _, row in metadata.iterrows()}

    expanded = []
    seen = set()

    for item in results:
        for candidate_id in [
            item.get("prev_chunk_id"),
            item.get("chunk_id"),
            item.get("next_chunk_id"),
        ]:
            if candidate_id and candidate_id in chunk_id_to_row and candidate_id not in seen:
                row = dict(chunk_id_to_row[candidate_id])
                row["expanded_from"] = item["chunk_id"]
                expanded.append(row)
                seen.add(candidate_id)

    return expanded


# =========================================================
# FINAL RETRIEVAL API
# =========================================================

def retrieve(query: str,
             top_k_vector: int = TOP_K_VECTOR,
             top_k_bm25: int = TOP_K_BM25,
             top_k_final: int = TOP_K_FINAL,
             expand_neighbors: bool = True,
             expand_full_blocks: bool = True) -> List[Dict]:

    vector_results = vector_search(query, top_k=top_k_vector)
    bm25_results = bm25_search(query, top_k=top_k_bm25)

    merged = merge_results(vector_results, bm25_results, top_k_final=top_k_final)

    final_items = merged

    if expand_full_blocks:
        final_items = expand_with_full_blocks(final_items)
    elif expand_neighbors:
        final_items = expand_with_neighbors(final_items)

    # dedup + tri cohérent
    dedup = {}
    for item in final_items:
        dedup[item["chunk_id"]] = item

    final_items = list(dedup.values())
    final_items = sorted(
        final_items,
        key=lambda x: (
            x.get("doc_id", ""),
            x.get("block_id", -1),
            x.get("chunk_nb", -1),
        )
    )

    return final_items

def get_chunk_ids(chunks: List[Dict]) -> List[str]:
    return [ch["chunk_id"] for ch in chunks]


# =========================================================
# CONTEXT BUILDER
# =========================================================

def build_context(chunks: List[Dict], max_chars: int = 12000) -> str:
    parts = []
    current_len = 0

    for ch in chunks:
        citation = (
            f"[chunk_id: {ch.get('chunk_id')}] "
            f"[doc: {ch.get('doc_title')}] "
            f"[title: {ch.get('title_nb') or '-'} {ch.get('title_name') or ''}] "
            f"[chapter: {ch.get('chapter_nb') or '-'} {ch.get('chapter_name') or ''}] "
            f"[section: {ch.get('section_nb') or '-'} {ch.get('section_name') or ''}] "
            f"[article: {ch.get('article_nb') or '-'} {ch.get('article_name') or ''}]"
        ).strip()

        text = ch.get("text_for_embedding", "") or ch.get("text", "")
        block = f"{citation}\n{text}\n"

        if current_len + len(block) > max_chars:
            break

        parts.append(block)
        current_len += len(block)

    return "\n".join(parts)


def build_context_payload(chunks: List[Dict], max_chars: int = 12000) -> Dict:
    parts = []
    chunk_ids = []
    current_len = 0

    for ch in chunks:
        citation = (
            f"[chunk_id: {ch.get('chunk_id')}] "
            f"[doc: {ch.get('doc_title')}] "
            f"[date: {ch.get('doc_date')}] "
            f"[title: {ch.get('title_nb') or '-'} {ch.get('title_name') or ''}] "
            f"[chapter: {ch.get('chapter_nb') or '-'} {ch.get('chapter_name') or ''}] "
            f"[section: {ch.get('section_nb') or '-'} {ch.get('section_name') or ''}] "
            f"[article: {ch.get('article_nb') or '-'} {ch.get('article_name') or ''}]"
        ).strip()

        text = ch.get("text_for_embedding", "") or ch.get("text", "")
        block = f"{citation}\n{text}\n"

        if current_len + len(block) > max_chars:
            break

        parts.append(block)
        chunk_ids.append(ch["chunk_id"])
        current_len += len(block)

    return {
        "context": "\n".join(parts),
        "chunk_ids": chunk_ids,
        "used_chunks": chunks[:len(chunk_ids)],
    }