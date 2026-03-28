import json
import os
import re
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from tools.fare_tool import FareCalculator
from tools.distance_tool import DistanceEstimator
from tools.route_optimizer import RouteOptimizer
from tools.traffic_tool import TrafficTool
from tools.weather_tool import WeatherTool
from tools.urban_trip_planner import UrbanTripPlanner


class ToolAgentFactory:

    def __init__(self, dataframe, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"):
        load_dotenv()

        self.fare_tool = FareCalculator(dataframe)
        self.distance_tool = DistanceEstimator(dataframe)
        self.route_optimizer = RouteOptimizer(dataframe)
        self.traffic_tool = TrafficTool(dataframe)
        self.weather_tool = WeatherTool(dataframe)
        self.urban_trip_planner = UrbanTripPlanner(dataframe)
        token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "Missing Hugging Face token. Set HUGGINGFACE_API_TOKEN (or HF_TOKEN) in .env."
            )
        self.client = InferenceClient(token=token)
        self.reasoning_model = model_name or os.getenv(
            "REASONING_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        )
        self.summarization_model = os.getenv("SUMMARIZATION_MODEL", self.reasoning_model)

    def create_agent(self):
        return _DeepSeekToolExecutor(
            fare_tool=self.fare_tool,
            distance_tool=self.distance_tool,
            route_optimizer=self.route_optimizer,
            traffic_tool=self.traffic_tool,
            weather_tool=self.weather_tool,
            urban_trip_planner=self.urban_trip_planner,
            client=self.client,
            reasoning_model=self.reasoning_model,
            summarization_model=self.summarization_model,
        )


class _DeepSeekToolExecutor:
    def __init__(
        self,
        fare_tool,
        distance_tool,
        route_optimizer,
        traffic_tool,
        weather_tool,
        urban_trip_planner,
        client: InferenceClient,
        reasoning_model: str,
        summarization_model: str,
    ):
        self.fare_tool = fare_tool
        self.distance_tool = distance_tool
        self.route_optimizer = route_optimizer
        self.traffic_tool = traffic_tool
        self.weather_tool = weather_tool
        self.urban_trip_planner = urban_trip_planner
        self.client = client
        self.reasoning_model = reasoning_model
        self.summarization_model = summarization_model

    def invoke(self, payload):
        user_text = payload.get("input", "")
        if not user_text.strip():
            return {"output": "Please provide a query."}

        plan = self._plan_tool_call(user_text)
        pickup_zone = plan.get("pickup_zone") or self._extract_route(user_text)[0]
        dropoff_zone = plan.get("dropoff_zone") or self._extract_route(user_text)[1]
        tool_name = plan.get("tool") or self._fallback_tool_name(user_text)

        if not pickup_zone or not dropoff_zone:
            return {"output": "Please specify both pickup and dropoff zones (e.g., 'from Midtown to JFK Airport')."}

        tool_map = {
            "fare_calculator": self.fare_tool.estimate,
            "distance_estimator": self.distance_tool.estimate,
            "route_optimizer": self.route_optimizer.optimize,
            "traffic_analyzer": self.traffic_tool.analyze,
            "weather_analyzer": self.weather_tool.analyze,
            "urban_trip_planner": self.urban_trip_planner.plan_trip,
        }
        tool_fn = tool_map.get(tool_name, self.fare_tool.estimate)
        result = tool_fn(pickup_zone, dropoff_zone)
        output_text = self._summarize_result(user_text, tool_name, pickup_zone, dropoff_zone, result)
        return {"output": output_text}

    def _chat(self, model_name: str, prompt: str, max_tokens: int = 220, temperature: float = 0.1) -> str:
        response = self.client.chat_completion(
            model=model_name,
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

    def _plan_tool_call(self, query: str):
        prompt = f"""
Return ONLY valid JSON with keys: tool, pickup_zone, dropoff_zone.
tool must be one of:
["urban_trip_planner","fare_calculator","distance_estimator","route_optimizer","traffic_analyzer","weather_analyzer"].
If unknown values, use empty strings.

Query: {query}
"""
        raw = self._chat(self.reasoning_model, prompt, max_tokens=180, temperature=0.0)
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
        return {}

    def _extract_route(self, query: str):
        q = query.strip().rstrip("?")
        match = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+)$", q, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        match = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+)$", q, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return "", ""

    def _fallback_tool_name(self, query: str):
        q = query.lower()
        if any(marker in q for marker in ("real-time", "realtime", "right now", "current", "traffic", "weather", "rain")):
            return "urban_trip_planner"
        if "estimate fare" in q or "how much would it cost" in q:
            return "fare_calculator"
        if "distance" in q or "how far" in q:
            return "distance_estimator"
        if "fastest route" in q or "shortest route" in q:
            return "route_optimizer"
        if "traffic" in q:
            return "traffic_analyzer"
        if "weather" in q:
            return "weather_analyzer"
        return "fare_calculator"

    def _summarize_result(self, query: str, tool_name: str, pickup_zone: str, dropoff_zone: str, result):
        if not isinstance(result, dict):
            return str(result)
        if result.get("status") == "error":
            return result.get("message", "Tool execution failed.")

        prompt = f"""
You are a precise assistant. Summarize the tool result in 2-4 short lines.
Use only the fields in the JSON. Do not invent values.
Mention pickup and dropoff zones.

User query: {query}
Tool used: {tool_name}
Pickup zone: {pickup_zone}
Dropoff zone: {dropoff_zone}
Tool JSON result:
{json.dumps(result, ensure_ascii=True)}
"""
        summary = self._chat(self.summarization_model, prompt, max_tokens=220, temperature=0.1)
        return summary or "\n".join(f"{k}: {v}" for k, v in result.items())
