import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure project-root imports work when running this file directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CSV_PATH


class FareCalculator:
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
            (self.df["PU_Zone"] == pickup_zone_resolved)
            & (self.df["DO_Zone"] == dropoff_zone_resolved)
        ]
        used_reverse_fallback = False

        if route_df.empty:
            reverse_df = self.df[
                (self.df["PU_Zone"] == dropoff_zone_resolved)
                & (self.df["DO_Zone"] == pickup_zone_resolved)
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

        median_fare = route_df["total_amount"].median()

        return {
            "status": "success",
            "pickup_zone": pickup_zone_resolved,
            "dropoff_zone": dropoff_zone_resolved,
            "estimated_fare": round(float(median_fare), 2),
            "confidence": "historical_median",
            "sample_size": len(route_df),
            "used_reverse_route_fallback": used_reverse_fallback,
        }


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Estimate historical taxi fare between pickup and dropoff zones."
    )
    parser.add_argument("--pickup", dest="pickup_zone", help="Pickup zone name")
    parser.add_argument("--dropoff", dest="dropoff_zone", help="Dropoff zone name")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    pickup = (args.pickup_zone or "").strip()
    dropoff = (args.dropoff_zone or "").strip()

    if not pickup:
        pickup = input("Pickup zone: ").strip()
    if not dropoff:
        dropoff = input("Dropoff zone: ").strip()

    if not pickup or not dropoff:
        print("Both pickup and dropoff zones are required.")
        raise SystemExit(1)

    df = pd.read_csv(
        CSV_PATH,
        parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"],
    )
    tool = FareCalculator(df)
    print(tool.estimate(pickup, dropoff))
