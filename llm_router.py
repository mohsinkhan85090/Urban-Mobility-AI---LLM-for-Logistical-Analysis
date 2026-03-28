import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from config import REASONING_MODEL

load_dotenv()

_TOOLS = [
    "zone_resolver",
    "distance_tool",
    "fare_tool",
    "route_optimizer",
    "urban_trip_planner",
]


@dataclass
class RouteDecision:
    intent: str
    confidence: float
    use_tool: bool
    selected_tool: str
    tool_args: Dict[str, Any]
    use_rag: bool
    rag_after_tool: bool
    entities: Dict[str, Any]
    reason: str


class LLMRouter:
    def __init__(self, confidence_threshold: float = 0.65):
        token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "Missing Hugging Face token. Set HUGGINGFACE_API_TOKEN (or HF_TOKEN) in .env."
            )
        self.client = InferenceClient(token=token)
        self.model = os.getenv("REASONING_MODEL", REASONING_MODEL)
        self.confidence_threshold = confidence_threshold

    def route(self, query: str) -> RouteDecision:
        llm_json = self._classify_with_llm(query)
        decision = self._build_decision(query, llm_json)

        # Force tool usage when confidence passes threshold for computational/hybrid
        if (
            decision.intent in {"COMPUTATIONAL", "HYBRID"}
            and decision.confidence >= self.confidence_threshold
            and not decision.use_tool
        ):
            decision.use_tool = True
            decision.selected_tool = self._keyword_tool(query)

        if decision.use_tool and not decision.selected_tool:
            decision.selected_tool = self._keyword_tool(query)

        if decision.use_tool:
            decision.tool_args = self._normalize_tool_args(
                query, decision.selected_tool, decision.tool_args, decision.entities
            )

        return decision

    def _classify_with_llm(self, query: str) -> Dict[str, Any]:
        prompt = f"""
You are a strict router for an NYC TLC assistant.
Choose whether the query is:
- ANALYTICAL (RAG over dataset),
- COMPUTATIONAL (must call tool),
- HYBRID (tool first, then RAG explanation).

Tool rules:
- zone_resolver: map zone/borough text to location IDs.
- distance_tool: distance, miles, "how far", "distance between".
- fare_tool: fare, cost, price, "how much", total amount.
- route_optimizer: best/fastest/optimal route, avoid traffic.
- urban_trip_planner: multi-step trip planning or "plan my trip".

Return ONLY JSON with keys:
intent, confidence, use_tool, selected_tool, tool_args, use_rag, rag_after_tool, entities, reason

Constraints:
- confidence is float [0,1].
- selected_tool must be one of {json.dumps(_TOOLS)} or "".
- tool_args must be an object.
- entities must include keys: pickup_zone, dropoff_zone, borough, datetime, passengers.

Query:
{query}
"""
        raw = self._chat(prompt, max_tokens=260, temperature=0.0)
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _build_decision(self, query: str, payload: Dict[str, Any]) -> RouteDecision:
        fallback_intent = self._keyword_intent(query)
        intent = str(payload.get("intent", fallback_intent)).upper()
        if intent not in {"ANALYTICAL", "COMPUTATIONAL", "HYBRID"}:
            intent = fallback_intent

        confidence = payload.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        selected_tool = str(payload.get("selected_tool", "")).strip()
        if selected_tool and selected_tool not in _TOOLS:
            selected_tool = ""

        entities = payload.get("entities", {})
        if not isinstance(entities, dict):
            entities = {}

        tool_args = payload.get("tool_args", {})
        if not isinstance(tool_args, dict):
            tool_args = {}

        use_tool = bool(payload.get("use_tool", intent in {"COMPUTATIONAL", "HYBRID"}))
        use_rag = bool(payload.get("use_rag", intent in {"ANALYTICAL", "HYBRID"}))
        rag_after_tool = bool(payload.get("rag_after_tool", intent == "HYBRID"))
        reason = str(payload.get("reason", "")).strip()

        if confidence < self.confidence_threshold:
            # Conservative fallback if LLM confidence is weak.
            intent = fallback_intent
            if intent == "ANALYTICAL":
                use_tool, use_rag, rag_after_tool = False, True, False
            elif intent == "COMPUTATIONAL":
                use_tool, use_rag, rag_after_tool = True, False, False
            else:
                use_tool, use_rag, rag_after_tool = True, True, True
            selected_tool = selected_tool or self._keyword_tool(query)
            reason = reason or "Fallback keyword routing due to low confidence."

        return RouteDecision(
            intent=intent,
            confidence=confidence,
            use_tool=use_tool,
            selected_tool=selected_tool,
            tool_args=tool_args,
            use_rag=use_rag,
            rag_after_tool=rag_after_tool,
            entities=entities,
            reason=reason,
        )

    def _normalize_tool_args(
        self,
        query: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        entities: Dict[str, Any],
    ) -> Dict[str, Any]:
        args = dict(tool_args)
        pickup_zone, dropoff_zone = self._extract_route(query)

        ent_pickup = str(entities.get("pickup_zone", "")).strip()
        ent_dropoff = str(entities.get("dropoff_zone", "")).strip()
        ent_borough = str(entities.get("borough", "")).strip()

        if tool_name == "zone_resolver":
            zone_or_borough = (
                str(args.get("zone_or_borough", "")).strip()
                or ent_borough
                or ent_pickup
                or ent_dropoff
            )
            if not zone_or_borough:
                zone_or_borough = self._extract_zone_like_value(query)
            return {"zone_or_borough": zone_or_borough}

        if tool_name in {"distance_tool", "fare_tool", "route_optimizer", "urban_trip_planner"}:
            return {
                "pickup_zone": str(args.get("pickup_zone", "")).strip() or ent_pickup or pickup_zone,
                "dropoff_zone": str(args.get("dropoff_zone", "")).strip() or ent_dropoff or dropoff_zone,
            }

        return args

    def _extract_route(self, query: str):
        q = query.strip().rstrip("?")
        match = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+)$", q, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        match = re.search(r"\bhow\s+far\s+is\s+(.+?)\s+from\s+(.+)$", q, flags=re.IGNORECASE)
        if match:
            # "How far is A from B" => route from B to A.
            return match.group(2).strip(), match.group(1).strip()
        match = re.search(r"\b(?:distance|miles)\s+(?:from\s+)?(.+?)\s+to\s+(.+)$", q, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        match = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+)$", q, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return "", ""

    @staticmethod
    def _extract_zone_like_value(query: str) -> str:
        q = query.strip().rstrip("?")
        for pattern in (r"\bof\s+(.+)$", r"\bfor\s+(.+)$", r"\bin\s+(.+)$"):
            match = re.search(pattern, q, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip(" '\"")
        return ""

    @staticmethod
    def _keyword_intent(query: str) -> str:
        q = query.lower()
        analytical_markers = [
            "trend",
            "compare",
            "comparison",
            "historical",
            "peak",
            "average fare at night",
            "which borough has highest fare",
            "analytics",
        ]
        computational_markers = [
            "how far",
            "distance",
            "miles",
            "how much",
            "fare",
            "cost",
            "price",
            "best route",
            "fastest way",
            "optimal path",
            "avoid traffic",
            "plan my trip",
            "suggest travel plan",
            "location id",
        ]
        is_analytical = any(m in q for m in analytical_markers)
        is_computational = any(m in q for m in computational_markers)

        if is_analytical and is_computational:
            return "HYBRID"
        if is_computational:
            return "COMPUTATIONAL"
        return "ANALYTICAL"

    @staticmethod
    def _keyword_tool(query: str) -> str:
        q = query.lower()
        if "location id" in q or "borough" in q:
            return "zone_resolver"
        if any(m in q for m in ("how far", "distance between", "miles from", "trip distance", "distance")):
            return "distance_tool"
        if any(m in q for m in ("how much", "estimated fare", "taxi cost", "total amount", "price from", "fare", "cost", "price")):
            return "fare_tool"
        if any(m in q for m in ("best route", "fastest way", "optimal path", "avoid traffic")):
            return "route_optimizer"
        if any(m in q for m in ("plan my trip", "suggest travel plan", "trip plan")):
            return "urban_trip_planner"
        return "fare_tool"

    def _chat(self, prompt: str, max_tokens: int = 220, temperature: float = 0.0) -> str:
        response = self.client.chat_completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
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
