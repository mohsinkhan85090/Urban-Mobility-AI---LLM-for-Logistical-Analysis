import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from embeddings import get_embeddings
from config import CSV_PATH, VECTOR_DB_DIR


def safe(val):
    return str(val) if pd.notna(val) else "Unknown"


def row_to_text(row):
    return f"""
    Pickup Location ID: {safe(row['PULocationID'])}
    Dropoff Location ID: {safe(row['DOLocationID'])}
    Pickup Zone: {safe(row['PU_Zone'])}
    Dropoff Zone: {safe(row['DO_Zone'])}
    Pickup Borough: {safe(row['PU_Borough'])}
    Dropoff Borough: {safe(row['DO_Borough'])}
    Distance: {safe(row['trip_distance'])} miles
    Total Fare: ${safe(row['total_amount'])}
    Airport Fee: ${safe(row['Airport_fee'])}
    Congestion Surcharge: ${safe(row['congestion_surcharge'])}
    """


def build_vector_store():
    df = pd.read_csv(CSV_PATH)

    docs = []

    for row in df.to_dict(orient="records"):
        docs.append(
            Document(
                page_content=row_to_text(row),
                metadata={
                    "pickup_location_id": safe(row["PULocationID"]),
                    "dropoff_location_id": safe(row["DOLocationID"]),
                    "pickup_zone": row["PU_Zone"],
                    "dropoff_zone": row["DO_Zone"],
                    "pickup_borough": row["PU_Borough"],
                    "dropoff_borough": row["DO_Borough"],
                }
            )
        )

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(VECTOR_DB_DIR)
    )

    print("Vector DB created successfully.")


if __name__ == "__main__":
    build_vector_store()
