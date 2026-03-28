import re
from typing import Any, Dict, Tuple

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from tools.distance_tool import DistanceEstimator
from tools.fare_tool import FareCalculator
from tools.route_optimizer import RouteOptimizer
from tools.urban_trip_planner import UrbanTripPlanner


class ZoneResolverInput(BaseModel):
    zone_or_borough: str = Field(..., min_length=1, description="NYC zone or borough text")


class RouteInput(BaseModel):
    pickup_zone: str = Field(..., min_length=1, description="Pickup zone name")
    dropoff_zone: str = Field(..., min_length=1, description="Dropoff zone name")


class ToolRegistry:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self._zone_ref = self._build_zone_ref(dataframe)
        self._tools = {
            "zone_resolver": {
                "description": "Map a zone/borough text into matching taxi zone IDs.",
                "schema": ZoneResolverInput,
                "runner": self._resolve_zone,
            },
            "distance_tool": {
                "description": "Compute historical average trip distance between pickup and dropoff zones.",
                "schema": RouteInput,
                "runner": DistanceEstimator(dataframe).estimate,
            },
            "fare_tool": {
                "description": "Compute historical median fare between pickup and dropoff zones.",
                "schema": RouteInput,
                "runner": FareCalculator(dataframe).estimate,
            },
            "route_optimizer": {
                "description": "Compute route-level duration/fare metrics for optimization.",
                "schema": RouteInput,
                "runner": RouteOptimizer(dataframe).optimize,
            },
            "urban_trip_planner": {
                "description": "Plan trip with traffic and weather adjusted fare/duration.",
                "schema": RouteInput,
                "runner": UrbanTripPlanner(dataframe).plan_trip,
            },
        }

    def tool_specs(self):
        specs = []
        for name, info in self._tools.items():
            schema = info["schema"]
            specs.append(
                {
                    "name": name,
                    "description": info["description"],
                    "schema": schema.model_json_schema(),
                }
            )
        return specs

    def execute(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        if tool_name not in self._tools:
            return {"status": "error", "message": f"Unsupported tool: {tool_name}"}, "TOOL_NOT_FOUND"

        spec = self._tools[tool_name]
        schema = spec["schema"]
        try:
            payload = schema(**(tool_args or {}))
        except ValidationError as exc:
            return {
                "status": "error",
                "message": "Invalid tool arguments",
                "details": exc.errors(),
            }, "INVALID_ARGUMENTS"

        args = payload.model_dump()
        try:
            result = spec["runner"](**args)
        except TypeError as exc:
            return {"status": "error", "message": f"Tool execution signature error: {exc}"}, "EXECUTION_ERROR"
        except Exception as exc:
            return {"status": "error", "message": f"Tool execution failed: {exc}"}, "EXECUTION_ERROR"

        if not isinstance(result, dict):
            return {"status": "error", "message": "Tool returned non-dict response"}, "BAD_TOOL_RESPONSE"

        if result.get("status") == "error":
            return result, "TOOL_ERROR"
        return result, "OK"

    @staticmethod
    def _build_zone_ref(df: pd.DataFrame) -> pd.DataFrame:
        pu = df[["PULocationID", "PU_Zone", "PU_Borough"]].rename(
            columns={"PULocationID": "LocationID", "PU_Zone": "Zone", "PU_Borough": "Borough"}
        )
        do = df[["DOLocationID", "DO_Zone", "DO_Borough"]].rename(
            columns={"DOLocationID": "LocationID", "DO_Zone": "Zone", "DO_Borough": "Borough"}
        )
        ref = pd.concat([pu, do], ignore_index=True).dropna()
        ref["Zone"] = ref["Zone"].astype(str).str.strip()
        ref["Borough"] = ref["Borough"].astype(str).str.strip()
        ref["LocationID"] = ref["LocationID"].astype(int)
        return ref.drop_duplicates()

    def _resolve_zone(self, zone_or_borough: str):
        text = re.sub(r"\s+", " ", zone_or_borough.strip())
        if not text:
            return {"status": "error", "message": "zone_or_borough is required."}

        exact_zone = self._zone_ref[self._zone_ref["Zone"].str.lower() == text.lower()]
        if not exact_zone.empty:
            rows = exact_zone.sort_values("LocationID").head(20)
            return {
                "status": "success",
                "query": text,
                "match_type": "zone_exact",
                "matches": rows[["LocationID", "Zone", "Borough"]].to_dict(orient="records"),
            }

        exact_borough = self._zone_ref[self._zone_ref["Borough"].str.lower() == text.lower()]
        if not exact_borough.empty:
            rows = exact_borough.sort_values(["Zone", "LocationID"]).head(50)
            return {
                "status": "success",
                "query": text,
                "match_type": "borough_exact",
                "matches": rows[["LocationID", "Zone", "Borough"]].to_dict(orient="records"),
            }

        fuzzy = self._zone_ref[
            self._zone_ref["Zone"].str.contains(text, case=False, na=False)
            | self._zone_ref["Borough"].str.contains(text, case=False, na=False)
        ]
        if fuzzy.empty:
            return {"status": "error", "message": f"No zone/borough match found for '{text}'."}

        rows = fuzzy.sort_values(["Zone", "LocationID"]).head(20)
        return {
            "status": "success",
            "query": text,
            "match_type": "fuzzy",
            "matches": rows[["LocationID", "Zone", "Borough"]].to_dict(orient="records"),
        }
