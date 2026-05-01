from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd


# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


# -------------------------------------------------
# Loading helpers
# -------------------------------------------------
def load_data() -> Dict[str, pd.DataFrame]:
    """Load all required CSV files from the data folder."""
    required_files = {
        "drivers": DATA_DIR / "drivers.csv",
        "passengers": DATA_DIR / "passengers.csv",
        "requests": DATA_DIR / "ride_requests.csv",
        "weights": DATA_DIR / "preference_weights.csv",
        "pricing": DATA_DIR / "driver_type_pricing.csv",
        "offers": DATA_DIR / "offers.csv",
    }

    missing = [str(path) for path in required_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The following required files are missing:\n" + "\n".join(missing)
        )

    return {
        "drivers": pd.read_csv(required_files["drivers"]),
        "passengers": pd.read_csv(required_files["passengers"]),
        "requests": pd.read_csv(required_files["requests"]),
        "weights": pd.read_csv(required_files["weights"]),
        "pricing": pd.read_csv(required_files["pricing"]),
        "offers": pd.read_csv(required_files["offers"]),
    }


# -------------------------------------------------
# Column normalization helpers
# -------------------------------------------------
def normalize_columns(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Normalize column names and add missing fields so the rest of the
    pipeline can run even if the CSV headers vary slightly.
    """
    drivers = data["drivers"].copy()
    passengers = data["passengers"].copy()
    requests = data["requests"].copy()
    weights = data["weights"].copy()
    pricing = data["pricing"].copy()
    offers = data["offers"].copy()

    # -------------------------
    # Normalize passenger columns
    # -------------------------
    passenger_rename_map = {
        "max_wait_time": "max_wait_time_min",
    }
    passengers.rename(columns=passenger_rename_map, inplace=True)

    # -------------------------
    # Normalize offer columns
    # -------------------------
    offer_rename_map = {
        "distance": "distance_to_pickup_km",
        "estimated_wait_time": "estimated_wait_time_min",
    }
    offers.rename(columns=offer_rename_map, inplace=True)

    # -------------------------
    # Required defaults
    # -------------------------
    if "offer_id" not in offers.columns:
        offers["offer_id"] = [f"O{i+1:04d}" for i in range(len(offers))]

    if "eligible_offer" not in offers.columns:
        offers["eligible_offer"] = "yes"

    if "recommendation_flag" not in offers.columns:
        offers["recommendation_flag"] = "no"

    if "budget_cap" not in passengers.columns:
        passengers["budget_cap"] = float("inf")

    if "preferred_vehicle" not in passengers.columns:
        passengers["preferred_vehicle"] = "any"

    if "party_size" not in passengers.columns:
        passengers["party_size"] = 1

    if "max_wait_time_min" not in passengers.columns:
        passengers["max_wait_time_min"] = float("inf")

    if "demand_level" not in requests.columns:
        requests["demand_level"] = "medium"

    if "weather" not in requests.columns:
        requests["weather"] = "clear"

    if "request_time" not in requests.columns:
        requests["request_time"] = list(range(1, len(requests) + 1))

    # -------------------------
    # Driver rating into offers
    # -------------------------
    if "driver_rating" not in offers.columns:
        if "rating" in drivers.columns:
            offers = offers.merge(
                drivers[["driver_id", "rating"]],
                on="driver_id",
                how="left",
            )
            offers.rename(columns={"rating": "driver_rating"}, inplace=True)
        else:
            offers["driver_rating"] = 0.0

    # -------------------------
    # Merge request_id if missing
    # -------------------------
    if "request_id" not in offers.columns:
        if "request_id" in requests.columns and "passenger_id" in requests.columns:
            offers = offers.merge(
                requests[["request_id", "passenger_id"]],
                on="passenger_id",
                how="left",
            )

    # -------------------------
    # Add score if missing
    # -------------------------
    if "score" not in offers.columns:
        offers["score"] = (
            0.5 * offers["offered_price"]
            + 0.3 * offers["estimated_wait_time_min"]
            - 0.2 * offers["driver_rating"]
        )

    return {
        "drivers": drivers,
        "passengers": passengers,
        "requests": requests,
        "weights": weights,
        "pricing": pricing,
        "offers": offers,
    }


# -------------------------------------------------
# Core graph model
# -------------------------------------------------
def build_bipartite_graph(
    drivers: pd.DataFrame,
    passengers: pd.DataFrame,
    offers: pd.DataFrame,
) -> nx.Graph:
    """Create a weighted bipartite graph from project data."""
    graph = nx.Graph()

    for _, row in drivers.iterrows():
        graph.add_node(
            row["driver_id"],
            bipartite=0,
            node_type="driver",
            rating=row.get("rating", None),
            driver_type=row.get("driver_type", None),
            availability=row.get("availability", None),
        )

    for _, row in passengers.iterrows():
        graph.add_node(
            row["passenger_id"],
            bipartite=1,
            node_type="passenger",
            preference=row.get("preference_type", None),
            budget_cap=row.get("budget_cap", None),
            max_wait_time_min=row.get("max_wait_time_min", None),
        )

    for _, row in offers.iterrows():
        graph.add_edge(
            row["driver_id"],
            row["passenger_id"],
            offer_id=row.get("offer_id", None),
            score=float(row["score"]),
            price=float(row["offered_price"]),
            wait_time=float(row["estimated_wait_time_min"]),
            distance=float(row["distance_to_pickup_km"]),
            eligible=row.get("eligible_offer", "yes"),
            recommendation_flag=row.get("recommendation_flag", "no"),
        )

    return graph


# -------------------------------------------------
# Offer processing
# -------------------------------------------------
def preprocess_offers(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all useful information into one clean offer table."""
    offers = data["offers"].copy()
    passengers = data["passengers"].copy()
    requests = data["requests"].copy()
    weights = data["weights"].copy()

    offers["eligible_bool"] = (
        offers["eligible_offer"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["yes", "true", "1"])
    )

    # Merge passenger fields
    passenger_cols = [
        "passenger_id",
        "preference_type",
        "budget_cap",
        "preferred_vehicle",
        "party_size",
        "max_wait_time_min",
    ]
    passenger_cols = [c for c in passenger_cols if c in passengers.columns]

    offers = offers.merge(
        passengers[passenger_cols],
        on="passenger_id",
        how="left",
        suffixes=("", "_from_passenger"),
    )

    # Merge request fields
    request_cols = [
        "request_id",
        "passenger_id",
        "request_time",
        "demand_level",
        "weather",
    ]
    request_cols = [c for c in request_cols if c in requests.columns]

    if "request_id" in offers.columns and "request_id" in requests.columns:
        offers = offers.merge(
            requests[[c for c in request_cols if c != "passenger_id"]],
            on="request_id",
            how="left",
        )
    elif "passenger_id" in offers.columns and "passenger_id" in requests.columns:
        offers = offers.merge(
            requests[request_cols],
            on="passenger_id",
            how="left",
        )

    # Merge preference weights if available
    if "preference_type" in weights.columns:
        offers = offers.merge(weights, on="preference_type", how="left")

    # Fill numeric defaults defensively
    offers["budget_cap"] = offers["budget_cap"].fillna(float("inf"))
    offers["max_wait_time_min"] = offers["max_wait_time_min"].fillna(float("inf"))

    # Derived flags
    offers["within_budget"] = offers["offered_price"] <= offers["budget_cap"]
    offers["within_wait_limit"] = (
        offers["estimated_wait_time_min"] <= offers["max_wait_time_min"]
    )
    offers["relaxed_eligible"] = offers["within_budget"] | offers["within_wait_limit"]

    return offers


# -------------------------------------------------
# Recommendation helpers
# -------------------------------------------------
def get_top_offers_per_passenger(offers: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Return top N recommended offers per passenger based on score."""
    ranked = offers.copy()
    ranked = ranked.sort_values(
        ["passenger_id", "score", "offered_price", "estimated_wait_time_min"]
    )
    ranked["rank_by_score"] = ranked.groupby("passenger_id").cumcount() + 1
    return ranked[ranked["rank_by_score"] <= top_n].copy()


# -------------------------------------------------
# Simulation strategies
# -------------------------------------------------
def choose_offer_for_passenger(
    passenger_offers: pd.DataFrame,
    preference_type: str,
) -> pd.Series:
    """Simulate passenger choice based on preference."""
    if passenger_offers.empty:
        raise ValueError("Passenger has no candidate offers.")

    preference_type = str(preference_type).strip().lower()

    if preference_type == "cheap":
        idx = passenger_offers["offered_price"].idxmin()
    elif preference_type == "fast":
        idx = passenger_offers["estimated_wait_time_min"].idxmin()
    elif preference_type == "premium":
        idx = passenger_offers["driver_rating"].idxmax()
    else:  # balanced/default
        idx = passenger_offers["score"].idxmin()

    return passenger_offers.loc[idx]


def sort_requests(passenger_order: pd.DataFrame) -> pd.DataFrame:
    """Return ride requests in stable processing order."""
    req = passenger_order.copy()

    if "request_time" not in req.columns:
        req["request_time"] = list(range(1, len(req) + 1))

    if "request_id" not in req.columns:
        req["request_id"] = [f"R{i+1:03d}" for i in range(len(req))]

    return req.sort_values(["request_time", "request_id"]).copy()


def simulate_marketplace_choice(
    offers: pd.DataFrame,
    passenger_order: pd.DataFrame,
    candidate_mode: str = "strict",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate ride assignment with unique-driver constraint.

    candidate_mode:
        - strict  : only eligible offers
        - relaxed : within budget OR within wait limit
        - all     : all offers
    """
    if candidate_mode == "strict":
        candidates = offers[offers["eligible_bool"]].copy()
    elif candidate_mode == "relaxed":
        candidates = offers[offers["relaxed_eligible"]].copy()
    elif candidate_mode == "all":
        candidates = offers.copy()
    else:
        raise ValueError("candidate_mode must be 'strict', 'relaxed', or 'all'.")

    assigned_drivers = set()
    selected_rows: List[pd.Series] = []
    unmatched_rows: List[dict] = []

    ordered_requests = sort_requests(passenger_order)

    for _, req in ordered_requests.iterrows():
        passenger_id = req["passenger_id"]

        p_offers = candidates[
            (candidates["passenger_id"] == passenger_id)
            & (~candidates["driver_id"].isin(assigned_drivers))
        ].copy()

        if p_offers.empty:
            unmatched_rows.append(
                {
                    "passenger_id": passenger_id,
                    "request_id": req.get("request_id", None),
                    "reason": f"No {candidate_mode} candidate offers available",
                }
            )
            continue

        preference_type = str(p_offers["preference_type"].iloc[0]).strip().lower()
        chosen = choose_offer_for_passenger(p_offers, preference_type)

        selected_rows.append(chosen)
        assigned_drivers.add(chosen["driver_id"])

    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    unmatched_df = pd.DataFrame(unmatched_rows)
    return selected_df, unmatched_df


def simulate_baseline_nearest_driver(
    offers: pd.DataFrame,
    passenger_order: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Baseline: each passenger gets nearest available driver."""
    assigned_drivers = set()
    selected_rows: List[pd.Series] = []
    unmatched_rows: List[dict] = []

    ordered_requests = sort_requests(passenger_order)

    for _, req in ordered_requests.iterrows():
        passenger_id = req["passenger_id"]

        p_offers = offers[
            (offers["passenger_id"] == passenger_id)
            & (~offers["driver_id"].isin(assigned_drivers))
        ].copy()

        if p_offers.empty:
            unmatched_rows.append(
                {
                    "passenger_id": passenger_id,
                    "request_id": req.get("request_id", None),
                    "reason": "No available drivers left",
                }
            )
            continue

        chosen = p_offers.loc[p_offers["distance_to_pickup_km"].idxmin()]
        selected_rows.append(chosen)
        assigned_drivers.add(chosen["driver_id"])

    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    unmatched_df = pd.DataFrame(unmatched_rows)
    return selected_df, unmatched_df


# -------------------------------------------------
# Evaluation
# -------------------------------------------------
def summarize_assignments(
    assignments: pd.DataFrame,
    unmatched: pd.DataFrame,
    strategy_name: str,
) -> pd.DataFrame:
    """Return one-row summary for a strategy."""
    if assignments.empty:
        return pd.DataFrame(
            [
                {
                    "strategy": strategy_name,
                    "matched_passengers": 0,
                    "unmatched_passengers": len(unmatched),
                    "avg_price": math.nan,
                    "avg_wait_time_min": math.nan,
                    "avg_pickup_distance_km": math.nan,
                    "avg_driver_rating": math.nan,
                    "total_price": math.nan,
                    "total_score": math.nan,
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "strategy": strategy_name,
                "matched_passengers": len(assignments),
                "unmatched_passengers": len(unmatched),
                "avg_price": round(assignments["offered_price"].mean(), 2),
                "avg_wait_time_min": round(assignments["estimated_wait_time_min"].mean(), 2),
                "avg_pickup_distance_km": round(assignments["distance_to_pickup_km"].mean(), 2),
                "avg_driver_rating": round(assignments["driver_rating"].mean(), 2),
                "total_price": round(assignments["offered_price"].sum(), 2),
                "total_score": round(assignments["score"].sum(), 2),
            }
        ]
    )


# -------------------------------------------------
# Output helpers
# -------------------------------------------------
def save_outputs(
    top_offers: pd.DataFrame,
    baseline_assignments: pd.DataFrame,
    baseline_unmatched: pd.DataFrame,
    strict_assignments: pd.DataFrame,
    strict_unmatched: pd.DataFrame,
    relaxed_assignments: pd.DataFrame,
    relaxed_unmatched: pd.DataFrame,
    all_assignments: pd.DataFrame,
    all_unmatched: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Save all output CSV files."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    top_offers.to_csv(OUTPUT_DIR / "top_3_offers_per_passenger.csv", index=False)
    baseline_assignments.to_csv(OUTPUT_DIR / "baseline_nearest_driver_assignments.csv", index=False)
    baseline_unmatched.to_csv(OUTPUT_DIR / "baseline_unmatched.csv", index=False)
    strict_assignments.to_csv(OUTPUT_DIR / "marketplace_strict_assignments.csv", index=False)
    strict_unmatched.to_csv(OUTPUT_DIR / "marketplace_strict_unmatched.csv", index=False)
    relaxed_assignments.to_csv(OUTPUT_DIR / "marketplace_relaxed_assignments.csv", index=False)
    relaxed_unmatched.to_csv(OUTPUT_DIR / "marketplace_relaxed_unmatched.csv", index=False)
    all_assignments.to_csv(OUTPUT_DIR / "marketplace_all_assignments.csv", index=False)
    all_unmatched.to_csv(OUTPUT_DIR / "marketplace_all_unmatched.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "strategy_summary.csv", index=False)


# -------------------------------------------------
# Main program
# -------------------------------------------------
def main() -> None:
    print("Loading CSV data...")
    raw_data = load_data()
    data = normalize_columns(raw_data)

    drivers = data["drivers"]
    passengers = data["passengers"]
    requests = data["requests"]
    offers = preprocess_offers(data)

    # Build graph from normalized offers
    graph = build_bipartite_graph(drivers, passengers, data["offers"])

    print("\n=== DATASET OVERVIEW ===")
    print(f"Drivers: {len(drivers)}")
    print(f"Passengers: {len(passengers)}")
    print(f"Requests: {len(requests)}")
    print(f"Offers: {len(offers)}")
    print(f"Graph nodes: {graph.number_of_nodes()}")
    print(f"Graph edges: {graph.number_of_edges()}")
    print(f"Strict eligible offers: {int(offers['eligible_bool'].sum())}")
    print(f"Relaxed eligible offers: {int(offers['relaxed_eligible'].sum())}")

    # Top recommendations
    top_offers = get_top_offers_per_passenger(offers, top_n=3)

    # Run strategies
    baseline_assignments, baseline_unmatched = simulate_baseline_nearest_driver(offers, requests)
    strict_assignments, strict_unmatched = simulate_marketplace_choice(offers, requests, "strict")
    relaxed_assignments, relaxed_unmatched = simulate_marketplace_choice(offers, requests, "relaxed")
    all_assignments, all_unmatched = simulate_marketplace_choice(offers, requests, "all")

    summary = pd.concat(
        [
            summarize_assignments(baseline_assignments, baseline_unmatched, "Baseline: nearest driver"),
            summarize_assignments(strict_assignments, strict_unmatched, "Marketplace: strict eligible only"),
            summarize_assignments(relaxed_assignments, relaxed_unmatched, "Marketplace: relaxed eligible"),
            summarize_assignments(all_assignments, all_unmatched, "Marketplace: all offers allowed"),
        ],
        ignore_index=True,
    )

    save_outputs(
        top_offers,
        baseline_assignments,
        baseline_unmatched,
        strict_assignments,
        strict_unmatched,
        relaxed_assignments,
        relaxed_unmatched,
        all_assignments,
        all_unmatched,
        summary,
    )

    print("\n=== STRATEGY SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== SAMPLE TOP 3 OFFERS PER PASSENGER ===")
    display_cols = [
        "passenger_id",
        "driver_id",
        "preference_type",
        "offered_price",
        "estimated_wait_time_min",
        "driver_rating",
        "score",
        "rank_by_score",
    ]
    existing_display_cols = [col for col in display_cols if col in top_offers.columns]
    print(top_offers[existing_display_cols].head(15).to_string(index=False))

    print("\nFiles saved in the 'outputs' folder.")


if __name__ == "__main__":
    main()

print("AWS pipeline test successful")
