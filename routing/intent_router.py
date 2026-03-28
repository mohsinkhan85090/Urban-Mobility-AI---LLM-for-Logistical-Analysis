# routing/intent_router.py

import re


class IntentRouter:

    def route(self, query: str) -> str:
        q = query.lower()
        realtime_markers = [
            "real-time",
            "realtime",
            "right now",
            "current conditions",
            "considering traffic",
            "considering weather",
            "traffic",
            "weather",
            "rain",
        ]

        if any(marker in q for marker in realtime_markers):
            return "REALTIME_TRIP_PLANNING"

        # Fare estimation
        if "estimate fare" in q or "how much would it cost" in q:
            return "ROUTE_FARE_ESTIMATE"

        # Route optimization
        if "fastest route" in q or "shortest route" in q:
            return "ROUTE_OPTIMIZATION"

        # Traffic impact
        if "traffic" in q:
            return "TRAFFIC_IMPACT"

        # Weather impact
        if "weather" in q:
            return "WEATHER_IMPACT"

        # Distance queries
        if "distance" in q:
            return "ROUTE_DISTANCE"

        # Location ID lookup
        if re.search(r"location id", q):
            return "LOCATION_LOOKUP"

        # Stats
        if "average fare" in q or "median fare" in q:
            return "DATA_STATS"

        return "GENERAL_RAG"
