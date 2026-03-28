class RouteOptimizer:

    def __init__(self, df):
        self.df = df
        from tools.zone_resolver import ZoneResolver
        self.resolver = ZoneResolver(df)

    def optimize(self, pickup_zone: str, dropoff_zone: str):
        pickup_zone_resolved = self.resolver.resolve(pickup_zone)
        dropoff_zone_resolved = self.resolver.resolve(dropoff_zone)

        if not pickup_zone_resolved or not dropoff_zone_resolved:
            return {
                "status": "error",
                "message": "Invalid zone name",
                "input_pickup_zone": pickup_zone,
                "input_dropoff_zone": dropoff_zone,
            }

        route_df = self.df[
            (self.df["PU_Zone"] == pickup_zone_resolved) &
            (self.df["DO_Zone"] == dropoff_zone_resolved)
        ]
        used_reverse_fallback = False

        if route_df.empty:
            reverse_df = self.df[
                (self.df["PU_Zone"] == dropoff_zone_resolved) &
                (self.df["DO_Zone"] == pickup_zone_resolved)
            ]
            if not reverse_df.empty:
                route_df = reverse_df
                used_reverse_fallback = True

        if route_df.empty:
            return {
                "status": "error",
                "message": "No historical data found for this route",
                "pickup_zone": pickup_zone_resolved,
                "dropoff_zone": dropoff_zone_resolved,
            }

        route_df = route_df.copy()
        route_df["duration"] = (
            route_df["tpep_dropoff_datetime"] -
            route_df["tpep_pickup_datetime"]
        ).dt.total_seconds() / 60

        fastest = route_df["duration"].mean()
        cheapest = route_df["total_amount"].mean()

        return {
            "status": "success",
            "pickup_zone": pickup_zone_resolved,
            "dropoff_zone": dropoff_zone_resolved,
            "avg_duration_minutes": round(float(fastest), 2),
            "avg_fare": round(float(cheapest), 2),
            "optimization_basis": "historical_average",
            "sample_size": len(route_df),
            "used_reverse_route_fallback": used_reverse_fallback,
        }
