import pandas as pd
import random
 

CSV_PATH = "./data/merged_taxi_data.csv"
OUTPUT_PATH = "generated_test_queries.csv"

# Load dataset
df = pd.read_csv(CSV_PATH)

# Extract unique zones
zones = list(set(df["PU_Zone"].dropna().unique()))
boroughs = list(set(df["PU_Borough"].dropna().unique()))

# Extract reasonable passenger counts
passenger_counts = [1, 2, 3, 4]

distance_templates = [
    "Distance between {z1} and {z2}?",
    "How far is {z1} from {z2}?",
    "Miles from {z1} to {z2}?"
]

fare_templates = [
    "How much is a taxi from {z1} to {z2}?",
    "Estimated fare from {z1} to {z2}?",
    "Taxi cost between {z1} and {z2}?"
]

route_templates = [
    "What is the fastest route from {z1} to {z2}?",
    "Best route from {z1} to {z2}?",
    "Optimal path between {z1} and {z2}?"
]

trip_templates = [
    "Plan a trip from {z1} to {z2} with {p} passengers.",
    "Suggest a travel plan from {z1} to {z2} for {p} people.",
    "Help me plan travel from {z1} to {z2} with {p} passengers."
]

def generate_queries(n=100):
    queries = []

    template_groups = [
        distance_templates,
        fare_templates,
        route_templates,
        trip_templates
    ]

    while len(queries) < n:
        z1, z2 = random.sample(zones, 2)
        templates = random.choice(template_groups)
        template = random.choice(templates)

        if "{p}" in template:
            p = random.choice(passenger_counts)
            query = template.format(z1=z1, z2=z2, p=p)
        else:
            query = template.format(z1=z1, z2=z2)

        queries.append(query)

    return queries

# Generate queries
queries = generate_queries(100)

# Save
pd.DataFrame({"query": queries}).to_csv(OUTPUT_PATH, index=False)

print(f"Generated {len(queries)} test queries → {OUTPUT_PATH}")