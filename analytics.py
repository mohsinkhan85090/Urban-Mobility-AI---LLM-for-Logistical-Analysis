import pandas as pd
from config import CSV_PATH
import re

df = pd.read_csv(CSV_PATH)
LOCATION_REF = df[["PULocationID", "PU_Zone", "PU_Borough"]].drop_duplicates()


def _extract_zone_from_query(query: str):
    q = query.strip().rstrip("?")
    patterns = [
        r"\bof\s+(.+)$",
        r"\bfor\s+(.+)$",
        r"\bfrom\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" '\"")
    return None


def resolve_location_id_query(query: str):
    q = query.lower()
    wants_pickup_id = "pulocationid" in q or "pickup location id" in q
    wants_dropoff_id = "dolocationid" in q or "dropoff location id" in q

    if not wants_pickup_id and not wants_dropoff_id:
        return None

    zone_text = _extract_zone_from_query(query)
    if not zone_text:
        return "Please specify a zone name, e.g. 'PULocationID of JFK Airport'."

    exact = LOCATION_REF[LOCATION_REF["PU_Zone"].astype(str).str.lower() == zone_text.lower()]
    if exact.empty:
        exact = LOCATION_REF[LOCATION_REF["PU_Zone"].astype(str).str.contains(zone_text, case=False, na=False)]

    if exact.empty:
        return f"No zone match found for '{zone_text}'."

    if len(exact) > 1:
        options = ", ".join(sorted(exact["PU_Zone"].astype(str).unique())[:10])
        return f"Multiple zone matches found: {options}. Please ask with an exact zone name."

    row = exact.iloc[0]
    zone = row["PU_Zone"]
    borough = row["PU_Borough"]
    loc_id = int(row["PULocationID"])

    if wants_pickup_id:
        return f"PULocationID for {zone} ({borough}) is {loc_id}."
    return (
        f"DOLocationID for {zone} ({borough}) is typically the same taxi zone id: {loc_id}. "
        "If you mean trip-specific dropoff rows, ask with route filters."
    )

def compute_statistics(retrieved_docs):
    pickup_zones = set(doc.metadata["pickup_zone"] for doc in retrieved_docs)
    dropoff_zones = set(doc.metadata["dropoff_zone"] for doc in retrieved_docs)

    filtered = df[
        df["PU_Zone"].isin(pickup_zones) &
        df["DO_Zone"].isin(dropoff_zones)
    ]

    if filtered.empty:
        return None

    return {
        "trip_count": len(filtered),
        "median_fare": round(filtered["total_amount"].median(), 2),
        "mean_fare": round(filtered["total_amount"].mean(), 2),
        "avg_distance": round(filtered["trip_distance"].mean(), 2),
    }
