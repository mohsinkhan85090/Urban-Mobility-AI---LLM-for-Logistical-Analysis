import os
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class TrafficService:
    """Google Distance Matrix wrapper with retries, timeouts, and safe fallbacks."""

    GOOGLE_DISTANCE_MATRIX_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_seconds: int = 8,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get_route_traffic(
        self,
        pickup_zone: str,
        dropoff_zone: str,
        historical_duration_minutes: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            return self._fallback(
                "GOOGLE_MAPS_API_KEY is not configured.",
                pickup_zone,
                dropoff_zone,
                historical_duration_minutes,
            )

        origin = f"{pickup_zone}, New York, NY"
        destination = f"{dropoff_zone}, New York, NY"
        params = {
            "origins": origin,
            "destinations": destination,
            "departure_time": "now",
            "traffic_model": "best_guess",
            "key": self.api_key,
        }

        try:
            response = self.session.get(
                self.GOOGLE_DISTANCE_MATRIX_URL,
                params=params,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            return self._fallback(
                f"Traffic API request failed: {exc}",
                pickup_zone,
                dropoff_zone,
                historical_duration_minutes,
            )
        except ValueError:
            return self._fallback(
                "Traffic API returned non-JSON data.",
                pickup_zone,
                dropoff_zone,
                historical_duration_minutes,
            )

        if payload.get("status") != "OK":
            return self._fallback(
                f"Traffic API status: {payload.get('status', 'UNKNOWN')}",
                pickup_zone,
                dropoff_zone,
                historical_duration_minutes,
            )

        rows = payload.get("rows", [])
        elements = rows[0].get("elements", []) if rows else []
        element = elements[0] if elements else {}
        if element.get("status") != "OK":
            return self._fallback(
                f"Traffic element status: {element.get('status', 'UNKNOWN')}",
                pickup_zone,
                dropoff_zone,
                historical_duration_minutes,
            )

        duration_in_traffic_sec = float(
            element.get("duration_in_traffic", {}).get("value", 0.0)
        )
        baseline_duration_sec = float(element.get("duration", {}).get("value", 0.0))
        distance_meters = float(element.get("distance", {}).get("value", 0.0))

        if duration_in_traffic_sec <= 0:
            return self._fallback(
                "Traffic API returned invalid duration_in_traffic.",
                pickup_zone,
                dropoff_zone,
                historical_duration_minutes,
            )

        duration_in_traffic_min = duration_in_traffic_sec / 60.0
        baseline_duration_min = (
            baseline_duration_sec / 60.0
            if baseline_duration_sec > 0
            else float(historical_duration_minutes or 0.0)
        )
        delay_ratio = 0.0
        if baseline_duration_min > 0:
            delay_ratio = max(
                0.0, (duration_in_traffic_min - baseline_duration_min) / baseline_duration_min
            )

        return {
            "status": "success",
            "provider": "google_distance_matrix",
            "origin": origin,
            "destination": destination,
            "distance_meters": round(distance_meters, 2),
            "baseline_duration_minutes": round(baseline_duration_min, 2),
            "duration_in_traffic_minutes": round(duration_in_traffic_min, 2),
            "traffic_delay_ratio": round(delay_ratio, 4),
            "is_fallback": False,
        }

    def _fallback(
        self,
        error_message: str,
        pickup_zone: str,
        dropoff_zone: str,
        historical_duration_minutes: Optional[float],
    ) -> Dict[str, Any]:
        if historical_duration_minutes and historical_duration_minutes > 0:
            value = round(float(historical_duration_minutes), 2)
            return {
                "status": "success",
                "provider": "historical_fallback",
                "origin": f"{pickup_zone}, New York, NY",
                "destination": f"{dropoff_zone}, New York, NY",
                "distance_meters": None,
                "baseline_duration_minutes": value,
                "duration_in_traffic_minutes": value,
                "traffic_delay_ratio": 0.0,
                "is_fallback": True,
                "api_error": error_message,
            }

        return {
            "status": "error",
            "provider": "google_distance_matrix",
            "message": error_message,
            "is_fallback": False,
        }
