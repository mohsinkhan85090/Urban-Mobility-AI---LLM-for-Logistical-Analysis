import os
import re
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from config import REASONING_MODEL, SUMMARIZATION_MODEL

load_dotenv()
_HF_CLIENT = None
ZONE_ALIASES = {
    "laguardia": ["laguardia airport"],
    "jfk": ["jfk airport"],
    "midtown": ["midtown center", "midtown east", "midtown north", "midtown south"],
}

def get_llm():
    global _HF_CLIENT
    if _HF_CLIENT is not None:
        return _HF_CLIENT

    token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Set HUGGINGFACE_API_TOKEN (or HF_TOKEN) in .env."
        )

    _HF_CLIENT = InferenceClient(token=token)
    return _HF_CLIENT


def _chat(model_name: str, prompt: str, max_tokens: int = 300) -> str:
    client = get_llm()
    response = client.chat_completion(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    content = response.choices[0].message.content
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        return "".join(parts).strip()
    return str(content).strip()

def _format_context(retrieved_docs):
    lines = []
    for i, doc in enumerate(retrieved_docs[:12], start=1):
        lines.append(f"[Doc {i}]")
        lines.append(doc.page_content.strip())
    return "\n".join(lines)


def _is_distance_query(query: str) -> bool:
    q = query.lower()
    return any(token in q for token in ("how far", "distance", "miles"))


def _extract_route(query: str):
    q = query.strip().rstrip("?")
    match = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+)$", q, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    match = re.search(r"\bis\s+(.+?)\s+to\s+(.+)$", q, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _expand_zone_candidates(zone: str):
    key = _normalize_text(zone)
    for alias_key, mapped in ZONE_ALIASES.items():
        if alias_key in key:
            return mapped
    return [key]


def _extract_doc_field(text: str, label: str):
    match = re.search(rf"{re.escape(label)}:\s*(.+)", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _extract_distance_from_doc(text: str):
    match = re.search(r"Distance:\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _build_distance_answer(query: str, retrieved_docs):
    if not _is_distance_query(query):
        return None
    route = _extract_route(query)
    if not route:
        return None

    origin, destination = route
    origin_candidates = set(_expand_zone_candidates(origin))
    destination_candidates = set(_expand_zone_candidates(destination))

    distances = []
    for doc in retrieved_docs:
        text = doc.page_content
        pickup = _normalize_text(_extract_doc_field(text, "Pickup Zone"))
        dropoff = _normalize_text(_extract_doc_field(text, "Dropoff Zone"))
        if pickup in origin_candidates and dropoff in destination_candidates:
            distance = _extract_distance_from_doc(text)
            if distance is not None:
                distances.append(distance)

    if not distances:
        return "I could not find that in the retrieved records."

    avg_distance = sum(distances) / len(distances)
    min_distance = min(distances)
    max_distance = max(distances)
    return (
        f"From {origin} to {destination}, the average trip distance is {avg_distance:.2f} miles "
        f"(min {min_distance:.2f}, max {max_distance:.2f}) based on {len(distances)} matching trips."
    )


def generate_rag_answer(query, retrieved_docs):
    distance_answer = _build_distance_answer(query, retrieved_docs)
    if distance_answer:
        return distance_answer

    context = _format_context(retrieved_docs)
    reasoning_model = os.getenv("REASONING_MODEL", REASONING_MODEL)
    summarization_model = os.getenv("SUMMARIZATION_MODEL", SUMMARIZATION_MODEL)

    reasoning_prompt = f"""
You are a strict RAG assistant for NYC taxi data.
Answer ONLY from the retrieved context below.
If the answer is not clearly present in the context, reply exactly:
"I could not find that in the retrieved records."
Do not use outside knowledge. Do not invent numbers or IDs.
When a query asks for an ID, return only the ID value and short label.

User Query:
{query}

Retrieved Context:
{context}
"""
    reasoning_text = _chat(reasoning_model, reasoning_prompt, max_tokens=600)

    summarization_prompt = f"""
Summarize the reasoning into a concise final answer for the user.
Keep it factual and grounded only in the provided reasoning/context.
If the reasoning indicates missing evidence, reply exactly:
"I could not find that in the retrieved records."

User Query:
{query}

Reasoning:
{reasoning_text}
"""

    final_text = _chat(summarization_model, summarization_prompt, max_tokens=220)
    return final_text.strip()
