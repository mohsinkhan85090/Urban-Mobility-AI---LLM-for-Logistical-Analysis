# routing/tool_registry.py

from tools.fare_tool import FareCalculator
from tools.distance_tool import DistanceEstimator
from tools.route_optimizer import RouteOptimizer
from tools.traffic_tool import TrafficTool
from tools.weather_tool import WeatherTool
from tools.urban_trip_planner import UrbanTripPlanner


class ToolRegistry:

    def __init__(self, dataframe):
        self.tools = {
            "fare_calculator": FareCalculator(dataframe),
            "distance_estimator": DistanceEstimator(dataframe),
            "route_optimizer": RouteOptimizer(dataframe),
            "traffic_tool": TrafficTool(dataframe),
            "weather_tool": WeatherTool(dataframe),
            "urban_trip_planner": UrbanTripPlanner(dataframe),
        }

    def execute(self, intent: str, params: dict):
        if intent == "ROUTE_FARE_ESTIMATE":
            return self.tools["fare_calculator"].estimate(**params)

        if intent == "ROUTE_DISTANCE":
            return self.tools["distance_estimator"].estimate(**params)

        if intent == "ROUTE_OPTIMIZATION":
            return self.tools["route_optimizer"].optimize(**params)

        if intent == "TRAFFIC_IMPACT":
            return self.tools["urban_trip_planner"].plan_trip(**params)

        if intent == "WEATHER_IMPACT":
            return self.tools["urban_trip_planner"].plan_trip(**params)

        if intent == "REALTIME_TRIP_PLANNING":
            return self.tools["urban_trip_planner"].plan_trip(**params)

        return None
