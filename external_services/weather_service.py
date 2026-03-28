import os
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class WeatherService:
    """OpenWeatherMap current weather wrapper with retries and safe handling."""

    OPEN_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_seconds: int = 8,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENWEATHERMAP_API_KEY")
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

    def get_current_weather(self, zone_name: str) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "status": "error",
                "provider": "openweathermap",
                "message": "OPENWEATHERMAP_API_KEY is not configured.",
            }

        params = {
            "q": f"{zone_name}, New York, US",
            "units": "metric",
            "appid": self.api_key,
        }

        try:
            response = self.session.get(
                self.OPEN_WEATHER_URL, params=params, timeout=self.timeout_seconds
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            return {
                "status": "error",
                "provider": "openweathermap",
                "message": f"Weather API request failed: {exc}",
            }
        except ValueError:
            return {
                "status": "error",
                "provider": "openweathermap",
                "message": "Weather API returned non-JSON data.",
            }

        condition = (
            (payload.get("weather") or [{}])[0].get("main")
            or (payload.get("weather") or [{}])[0].get("description")
            or "Unknown"
        )
        rain_intensity = float((payload.get("rain") or {}).get("1h", 0.0))
        visibility_m = float(payload.get("visibility", 10000))
        wind_speed_mps = float((payload.get("wind") or {}).get("speed", 0.0))

        weather_multiplier = self._compute_weather_multiplier(
            condition=condition,
            rain_intensity_mm_per_hr=rain_intensity,
            visibility_meters=visibility_m,
            wind_speed_mps=wind_speed_mps,
        )
        safety_delay_buffer_ratio = self._compute_safety_delay_buffer(visibility_m)

        return {
            "status": "success",
            "provider": "openweathermap",
            "zone": zone_name,
            "weather_condition": condition,
            "rain_intensity_mm_per_hr": round(rain_intensity, 2),
            "visibility_meters": round(visibility_m, 2),
            "wind_speed_mps": round(wind_speed_mps, 2),
            "weather_multiplier": round(weather_multiplier, 4),
            "safety_delay_buffer_ratio": round(safety_delay_buffer_ratio, 4),
        }

    @staticmethod
    def _compute_safety_delay_buffer(visibility_meters: float) -> float:
        if visibility_meters < 2000:
            return 0.10
        if visibility_meters < 5000:
            return 0.05
        return 0.0

    def _compute_weather_multiplier(
        self,
        condition: str,
        rain_intensity_mm_per_hr: float,
        visibility_meters: float,
        wind_speed_mps: float,
    ) -> float:
        multiplier = 1.0
        condition_normalized = (condition or "").strip().lower()

        if rain_intensity_mm_per_hr >= 7.0:
            multiplier += 0.18
        elif rain_intensity_mm_per_hr >= 2.5:
            multiplier += 0.10
        elif rain_intensity_mm_per_hr > 0:
            multiplier += 0.04
        elif "rain" in condition_normalized or "storm" in condition_normalized:
            multiplier += 0.08

        if visibility_meters < 2000:
            multiplier += 0.12
        elif visibility_meters < 5000:
            multiplier += 0.06

        if wind_speed_mps >= 12:
            multiplier += 0.07
        elif wind_speed_mps >= 8:
            multiplier += 0.03

        return multiplier
