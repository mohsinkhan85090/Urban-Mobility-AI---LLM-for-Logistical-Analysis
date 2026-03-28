from langchain_chroma import Chroma
from embeddings import get_embeddings
from config import VECTOR_DB_DIR, TOP_K
import re

ZONE_ALIASES = {
    "laguardia": ["LaGuardia Airport"],
    "jfk": ["JFK Airport"],
    "midtown": ["Midtown Center", "Midtown East", "Midtown North", "Midtown South"],
}


def _get_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings
    )


def _extract_route(query: str):
    q = query.strip().rstrip("?")

    match = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+)$", q, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    match = re.search(r"\bis\s+(.+?)\s+to\s+(.+)$", q, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    return None


def _merge_unique_docs(*doc_groups):
    merged = []
    seen = set()
    for group in doc_groups:
        for doc in group:
            key = doc.page_content.strip()
            if key not in seen:
                seen.add(key)
                merged.append(doc)
    return merged


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _expand_zone_candidates(zone: str):
    zone_clean = zone.strip()
    key = _normalize_text(zone_clean)
    for alias_key, mapped in ZONE_ALIASES.items():
        if alias_key in key:
            return mapped
    return [zone_clean]


def _extract_doc_zone(doc_text: str, label: str):
    match = re.search(rf"{re.escape(label)}:\s*(.+)", doc_text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def get_retriever():
    vectorstore = _get_vectorstore()

    return vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )


def retrieve_docs(query: str):
    vectorstore = _get_vectorstore()
    base_docs = vectorstore.similarity_search(query, k=TOP_K)

    route = _extract_route(query)
    if not route:
        return base_docs

    origin, destination = route
    origin_candidates = _expand_zone_candidates(origin)
    destination_candidates = _expand_zone_candidates(destination)

    exact_route_docs = []
    for origin_candidate in origin_candidates:
        for destination_candidate in destination_candidates:
            matches = vectorstore.similarity_search(
                "Distance for the exact route",
                k=max(TOP_K, 20),
                filter={
                    "$and": [
                        {"pickup_zone": origin_candidate},
                        {"dropoff_zone": destination_candidate},
                    ]
                },
            )
            exact_route_docs.extend(matches)

    route_query = (
        f"Pickup Zone: {' OR '.join(origin_candidates)}\n"
        f"Dropoff Zone: {' OR '.join(destination_candidates)}\n"
        "Distance:"
    )
    route_docs = vectorstore.similarity_search(route_query, k=max(TOP_K, 80))

    origin_tokens = {_normalize_text(x) for x in origin_candidates}
    destination_tokens = {_normalize_text(x) for x in destination_candidates}
    focused_docs = [
        d for d in route_docs
        if _normalize_text(_extract_doc_zone(d.page_content, "Pickup Zone")) in origin_tokens
        and _normalize_text(_extract_doc_zone(d.page_content, "Dropoff Zone")) in destination_tokens
    ]

    merged = _merge_unique_docs(exact_route_docs, focused_docs, route_docs, base_docs)
    return merged[:max(TOP_K, 80)]
