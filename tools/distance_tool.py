class DistanceEstimator:

    def __init__(self, df):
        self.df = df
        from tools.zone_resolver import ZoneResolver
        self.resolver = ZoneResolver(df)

    def estimate(self, pickup_zone: str, dropoff_zone: str):
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

        avg_distance = route_df["trip_distance"].mean()
        return {
            "status": "success",
            "pickup_zone": pickup_zone_resolved,
            "dropoff_zone": dropoff_zone_resolved,
            "distance_miles": round(float(avg_distance), 2),
            "confidence": "dataset_average",
            "sample_size": len(route_df),
            "used_reverse_route_fallback": used_reverse_fallback,
        }
