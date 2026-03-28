from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import pandas as pd

from external_services.traffic_service import TrafficService
from external_services.weather_service import WeatherService
from tools.distance_tool import DistanceEstimator
from tools.fare_tool import FareCalculator
from tools.zone_resolver import ZoneResolver


class UrbanTripPlanner:
    """Real-time trip planner combining historical data with live traffic/weather."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.fare_calculator = FareCalculator(df)
        self.distance_estimator = DistanceEstimator(df)
        self.zone_resolver = ZoneResolver(df)
        self.traffic_service = TrafficService()
        self.weather_service = WeatherService()

    def plan_trip(self, pickup_zone: str, dropoff_zone: str) -> Dict[str, Any]:
        pickup_zone_resolved = self.zone_resolver.resolve(pickup_zone)
        dropoff_zone_resolved = self.zone_resolver.resolve(dropoff_zone)
        if not pickup_zone_resolved or not dropoff_zone_resolved:
            return {
                "status": "error",
                "message": "Invalid zone name",
                "input_pickup_zone": pickup_zone,
                "input_dropoff_zone": dropoff_zone,
            }

        fare_result = self.fare_calculator.estimate(pickup_zone_resolved, dropoff_zone_resolved)
        if fare_result.get("status") != "success":
            return {
                "status": "error",
                "message": "Unable to compute historical fare.",
                "details": fare_result,
            }

        distance_result = self.distance_estimator.estimate(
            pickup_zone_resolved, dropoff_zone_resolved
        )
        if distance_result.get("status") != "success":
            return {
                "status": "error",
                "message": "Unable to compute historical distance.",
                "details": distance_result,
            }

        historical_duration = self._estimate_historical_duration_minutes(
            pickup_zone_resolved, dropoff_zone_resolved
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            traffic_future = executor.submit(
                self.traffic_service.get_route_traffic,
                pickup_zone_resolved,
                dropoff_zone_resolved,
                historical_duration,
            )
            weather_future = executor.submit(
                self.weather_service.get_current_weather, pickup_zone_resolved
            )
            traffic_result = traffic_future.result()
            weather_result = weather_future.result()

        if traffic_result.get("status") != "success":
            return {
                "status": "error",
                "message": "Unable to compute traffic-adjusted trip metrics.",
                "details": traffic_result,
            }

        if weather_result.get("status") != "success":
            weather_result = self._neutral_weather_fallback(weather_result.get("message"))

        traffic_delay_ratio = float(traffic_result.get("traffic_delay_ratio", 0.0))
        traffic_multiplier = 1.0
        if traffic_delay_ratio > 0.20:
            traffic_multiplier += traffic_delay_ratio

        weather_multiplier = float(weather_result.get("weather_multiplier", 1.0))
        safety_delay_buffer_ratio = float(weather_result.get("safety_delay_buffer_ratio", 0.0))

        base_fare = float(fare_result["estimated_fare"])
        traffic_duration_min = float(traffic_result["duration_in_traffic_minutes"])
        adjusted_duration_min = traffic_duration_min * (1.0 + safety_delay_buffer_ratio)
        final_estimated_fare = base_fare * traffic_multiplier * weather_multiplier

        confidence = self._resolve_confidence(
            fare_sample_size=int(fare_result.get("sample_size", 0)),
            distance_sample_size=int(distance_result.get("sample_size", 0)),
            used_traffic_fallback=bool(traffic_result.get("is_fallback", False)),
            used_weather_fallback=bool(weather_result.get("is_fallback", False)),
        )

        return {
            "status": "success",
            "pickup_zone": pickup_zone_resolved,
            "dropoff_zone": dropoff_zone_resolved,
            "base_fare": round(base_fare, 2),
            "historical_distance_miles": round(float(distance_result["distance_miles"]), 2),
            "traffic_adjusted_duration_minutes": round(adjusted_duration_min, 2),
            "weather_condition": weather_result.get("weather_condition", "Unknown"),
            "final_estimated_fare": round(final_estimated_fare, 2),
            "adjustment_breakdown": {
                "traffic_multiplier": round(traffic_multiplier, 4),
                "weather_multiplier": round(weather_multiplier, 4),
                "safety_delay_buffer_ratio": round(safety_delay_buffer_ratio, 4),
                "traffic_delay_ratio": round(traffic_delay_ratio, 4),
            },
            "realtime_context": {
                "traffic_provider": traffic_result.get("provider"),
                "weather_provider": weather_result.get("provider"),
                "rain_intensity_mm_per_hr": weather_result.get("rain_intensity_mm_per_hr", 0.0),
                "visibility_meters": weather_result.get("visibility_meters", 10000.0),
                "wind_speed_mps": weather_result.get("wind_speed_mps", 0.0),
            },
            "confidence": confidence,
        }

    def _estimate_historical_duration_minutes(
        self, pickup_zone_resolved: str, dropoff_zone_resolved: str
    ) -> Optional[float]:
        route_df = self.df[
            (self.df["PU_Zone"] == pickup_zone_resolved)
            & (self.df["DO_Zone"] == dropoff_zone_resolved)
        ].copy()
        if route_df.empty:
            return None

        durations = (
            route_df["tpep_dropoff_datetime"] - route_df["tpep_pickup_datetime"]
        ).dt.total_seconds() / 60.0
        durations = durations[(durations > 0) & (durations < 300)]
        if durations.empty:
            return None
        return float(durations.median())

    @staticmethod
    def _neutral_weather_fallback(error_message: Optional[str]) -> Dict[str, Any]:
        return {
            "status": "success",
            "provider": "historical_fallback",
            "weather_condition": "Unknown",
            "rain_intensity_mm_per_hr": 0.0,
            "visibility_meters": 10000.0,
            "wind_speed_mps": 0.0,
            "weather_multiplier": 1.0,
            "safety_delay_buffer_ratio": 0.0,
            "is_fallback": True,
            "api_error": error_message or "Weather API unavailable.",
        }

    @staticmethod
    def _resolve_confidence(
        fare_sample_size: int,
        distance_sample_size: int,
        used_traffic_fallback: bool,
        used_weather_fallback: bool,
    ) -> str:
        if used_traffic_fallback and used_weather_fallback:
            return "low"
        if fare_sample_size < 20 or distance_sample_size < 20:
            return "medium"
        if used_traffic_fallback or used_weather_fallback:
            return "medium"
        return "high"
