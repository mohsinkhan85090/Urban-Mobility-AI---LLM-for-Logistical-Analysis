import json
import logging
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from config import CSV_PATH, SUMMARIZATION_MODEL
from llm_router import LLMRouter
from taxi_rag_basic import handle_rag_query
from tool_registry import ToolRegistry

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("ask")

_DF_CACHE = None
_TOOL_REGISTRY = None
_ROUTER = None
_SUMMARIZER = None


def _get_df():
    global _DF_CACHE
    if _DF_CACHE is None:
        _DF_CACHE = pd.read_csv(
            CSV_PATH,
            parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"],
        )
    return _DF_CACHE


def _get_router():
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = LLMRouter(confidence_threshold=0.65)
    return _ROUTER


def _get_tool_registry():
    global _TOOL_REGISTRY
    if _TOOL_REGISTRY is None:
        _TOOL_REGISTRY = ToolRegistry(_get_df())
    return _TOOL_REGISTRY


def _get_summarizer():
    global _SUMMARIZER
    if _SUMMARIZER is not None:
        return _SUMMARIZER
    token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        return None
    _SUMMARIZER = InferenceClient(token=token)
    return _SUMMARIZER


def _summarize_tool_result(user_query: str, tool_name: str, tool_result: dict) -> str:
    client = _get_summarizer()
    if client is None:
        return json.dumps(tool_result, ensure_ascii=True)

    model_name = os.getenv("SUMMARIZATION_MODEL", SUMMARIZATION_MODEL)
    prompt = f"""
Summarize this tool output for the user in <=4 lines.
Do not invent values. Use only JSON fields.
Return final answer only. Do not include internal reasoning, thinking tags, or scratch work.

User query: {user_query}
Tool used: {tool_name}
Tool JSON:
{json.dumps(tool_result, ensure_ascii=True)}
"""
    try:
        response = client.chat_completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220,
            temperature=0.1,
        )
        content = response.choices[0].message.content
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            summary = "".join(parts).strip()
        else:
            summary = str(content).strip()
        summary = _clean_model_output(summary)
        return summary or json.dumps(tool_result, ensure_ascii=True)
    except Exception:
        return json.dumps(tool_result, ensure_ascii=True)


def _clean_model_output(text: str) -> str:
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    cleaned = re.sub(r"(?im)^\s*(thinking|reasoning)\s*:.*$", "", cleaned).strip()
    return cleaned


def handle_query(query: str) -> str:
    started = time.perf_counter()
    router = _get_router()
    registry = _get_tool_registry()

    decision = router.route(query)
    LOGGER.info(
        "route_decision intent=%s confidence=%.3f selected_tool=%s tool_args=%s use_tool=%s use_rag=%s rag_after_tool=%s",
        decision.intent,
        decision.confidence,
        decision.selected_tool,
        json.dumps(decision.tool_args, ensure_ascii=True),
        decision.use_tool,
        decision.use_rag,
        decision.rag_after_tool,
    )

    if decision.use_tool:
        tool_started = time.perf_counter()
        result, status = registry.execute(decision.selected_tool, decision.tool_args)
        tool_ms = (time.perf_counter() - tool_started) * 1000.0
        LOGGER.info(
            "tool_execution status=%s selected_tool=%s execution_time_ms=%.2f",
            status,
            decision.selected_tool,
            tool_ms,
        )

        # Computational paths must not hallucinate if tools fail.
        if status != "OK":
            return f"Tool execution failed: {result.get('message', 'unknown error')}"

        tool_summary = _summarize_tool_result(query, decision.selected_tool, result)

        if decision.rag_after_tool or (decision.intent == "HYBRID" and decision.use_rag):
            rag_started = time.perf_counter()
            rag_text = handle_rag_query(query)
            rag_ms = (time.perf_counter() - rag_started) * 1000.0
            LOGGER.info("rag_execution execution_time_ms=%.2f", rag_ms)
            total_ms = (time.perf_counter() - started) * 1000.0
            LOGGER.info("query_total execution_time_ms=%.2f", total_ms)
            return (
                f"Tool Result ({decision.selected_tool}):\n{tool_summary}\n\n"
                f"RAG Explanation:\n{rag_text}"
            )

        total_ms = (time.perf_counter() - started) * 1000.0
        LOGGER.info("query_total execution_time_ms=%.2f", total_ms)
        return f"Tool Result ({decision.selected_tool}):\n{tool_summary}"

    rag_started = time.perf_counter()
    rag_text = handle_rag_query(query)
    rag_ms = (time.perf_counter() - rag_started) * 1000.0
    total_ms = (time.perf_counter() - started) * 1000.0
    LOGGER.info("rag_execution execution_time_ms=%.2f", rag_ms)
    LOGGER.info("query_total execution_time_ms=%.2f", total_ms)
    return rag_text


if __name__ == "__main__":
    print("NYC Taxi Assistant (type 'exit' to quit)")
    while True:
        user_query = input("> ").strip()
        if user_query.lower() in {"exit", "quit"}:
            break
        if not user_query:
            continue
        try:
            print(handle_query(user_query))
        except Exception as exc:
            print(f"Error: {exc}")
