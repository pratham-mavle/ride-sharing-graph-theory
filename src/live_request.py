import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# ----------------------------
# Location Mapping
# ----------------------------
LOCATION_MAP = {
    "denton": (10, 20),
    "dallas": (50, 60),
    "plano": (70, 80),
    "frisco": (30, 40),
    "irving": (60, 30)
}

# ----------------------------
# Vehicle pricing multipliers
# ----------------------------
VEHICLE_OPTIONS = {
    "economy": 1.00,
    "standard": 1.15,
    "premium": 1.35
}


def distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def load_drivers():
    """Load drivers dataset."""
    file_path = DATA_DIR / "drivers.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {file_path}")
    return pd.read_csv(file_path)


def is_driver_available(value):
    """Treat common values as available."""
    value = str(value).strip().lower()
    return value in {"yes", "available", "true", "1"}


def arrival_category(wait_time):
    """Categorize wait time."""
    if wait_time <= 15:
        return "Fast"
    elif wait_time <= 30:
        return "Moderate"
    return "Slow"


def driver_minimum_acceptable_price(base_price, driver_rating, vehicle_type):
    """
    Minimum price a driver is willing to accept.
    Higher-rated and premium rides may require a slightly higher minimum.
    """
    vehicle_multiplier = VEHICLE_OPTIONS.get(vehicle_type, 1.0)
    min_price = (base_price * 0.9) + (driver_rating * 1.2) + (2.0 * vehicle_multiplier)
    return round(min_price, 2)


def generate_offers(drivers, pickup_x, pickup_y, drop_x, drop_y, vehicle_type):
    """
    Generate offers for all available drivers.
    Adds driver acceptance and recommendation details.
    """
    offers = []

    trip_distance = distance(pickup_x, pickup_y, drop_x, drop_y)
    vehicle_multiplier = VEHICLE_OPTIONS.get(vehicle_type, 1.0)

    for _, d in drivers.iterrows():
        if not is_driver_available(d.get("availability", "available")):
            continue

        driver_dist = distance(
            d["x_coordinate"],
            d["y_coordinate"],
            pickup_x,
            pickup_y
        )

        wait_time = driver_dist * 2

        # Reduced classroom-demo pricing so lower budgets can still see multiple offers
        raw_price = (
            d["base_price"]
            + (driver_dist * 0.25)
            + (trip_distance * 0.45)
        ) * vehicle_multiplier

        price = round(raw_price, 2)
        min_accept_price = driver_minimum_acceptable_price(
            d["base_price"], d["rating"], vehicle_type
        )

        score = round((0.5 * price) + (0.3 * wait_time) - (0.2 * d["rating"]), 2)

        offers.append({
            "driver_id": d["driver_id"],
            "driver_x": d["x_coordinate"],
            "driver_y": d["y_coordinate"],
            "pickup_distance": round(driver_dist, 2),
            "trip_distance": round(trip_distance, 2),
            "wait_time": round(wait_time, 2),
            "arrival_category": arrival_category(wait_time),
            "price": price,
            "rating": round(d["rating"], 2),
            "score": score,
            "vehicle_type": vehicle_type,
            "driver_min_accept_price": min_accept_price
        })

    return pd.DataFrame(offers)


def rank_offers(df, preference):
    """Rank offers based on user preference."""
    if df.empty:
        return df

    preference = preference.strip().lower()

    if preference == "cheap":
        return df.sort_values(["price", "wait_time", "score"])
    elif preference == "fast":
        return df.sort_values(["wait_time", "price", "score"])
    elif preference == "premium":
        return df.sort_values(["rating", "score"], ascending=[False, True])
    else:  # balanced/default
        return df.sort_values(["score", "price", "wait_time"])


def add_recommendation_tags(df):
    """Add simple recommendation tags to the filtered set."""
    if df.empty:
        return df

    tagged = df.copy()
    tagged["tag"] = ""

    cheapest_idx = tagged["price"].idxmin()
    fastest_idx = tagged["wait_time"].idxmin()
    highest_rated_idx = tagged["rating"].idxmax()
    best_overall_idx = tagged["score"].idxmin()

    tag_map = {
        cheapest_idx: "Cheapest Ride",
        fastest_idx: "Fastest Pickup",
        highest_rated_idx: "Highest Rated",
        best_overall_idx: "Best Overall"
    }

    for idx, tag in tag_map.items():
        if tagged.at[idx, "tag"] == "":
            tagged.at[idx, "tag"] = tag
        else:
            tagged.at[idx, "tag"] += f", {tag}"

    return tagged


def explain_qualification(row, passenger_min_budget, passenger_max_budget, preference):
    """Return a short reason explaining why the driver qualifies."""
    reasons = []

    if row["price"] >= passenger_min_budget and row["price"] <= passenger_max_budget:
        reasons.append("Within budget")

    if row["price"] >= row["driver_min_accept_price"]:
        reasons.append("Driver accepts fare")

    if row["arrival_category"] == "Fast":
        reasons.append("Fast arrival")

    if row["rating"] >= 4.8:
        reasons.append("High rating")

    preference = preference.lower()
    if preference == "cheap":
        reasons.append("Fits cheap preference")
    elif preference == "fast":
        reasons.append("Fits fast preference")
    elif preference == "premium":
        reasons.append("Fits premium preference")
    else:
        reasons.append("Fits balanced preference")

    return ", ".join(reasons)


def get_user_trip_input():
    """Get pickup/drop input either by coordinates or by location names."""
    print("\nChoose input method:")
    print("1. Enter coordinates")
    print("2. Enter location names")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        pickup_x = float(input("Enter pickup X: "))
        pickup_y = float(input("Enter pickup Y: "))
        drop_x = float(input("Enter drop X: "))
        drop_y = float(input("Enter drop Y: "))
        return pickup_x, pickup_y, drop_x, drop_y, None, None

    elif choice == "2":
        print("\nAvailable locations:", ", ".join(LOCATION_MAP.keys()))

        pickup = input("Enter pickup location: ").strip().lower()
        drop = input("Enter drop location: ").strip().lower()

        if pickup not in LOCATION_MAP or drop not in LOCATION_MAP:
            print("\nInvalid location entered.")
            print("Available locations are:", ", ".join(LOCATION_MAP.keys()))
            return None

        pickup_x, pickup_y = LOCATION_MAP[pickup]
        drop_x, drop_y = LOCATION_MAP[drop]

        return pickup_x, pickup_y, drop_x, drop_y, pickup, drop

    else:
        print("\nInvalid choice. Please enter 1 or 2.")
        return None


def format_display_table(df):
    """Format output to look cleaner and more app-like."""
    pretty = df.copy()

    pretty["pickup_distance"] = pretty["pickup_distance"].apply(lambda x: f"{x:.2f} km")
    pretty["trip_distance"] = pretty["trip_distance"].apply(lambda x: f"{x:.2f} km")
    pretty["wait_time"] = pretty["wait_time"].apply(lambda x: f"{x:.1f} min")
    pretty["price"] = pretty["price"].apply(lambda x: f"${x:.2f}")
    pretty["rating"] = pretty["rating"].apply(lambda x: f"{x:.1f} / 5")
    pretty["score"] = pretty["score"].apply(lambda x: f"{x:.2f}")
    pretty["driver_min_accept_price"] = pretty["driver_min_accept_price"].apply(lambda x: f"${x:.2f}")

    display_cols = [
        "driver_id",
        "vehicle_type",
        "pickup_distance",
        "trip_distance",
        "wait_time",
        "arrival_category",
        "price",
        "driver_min_accept_price",
        "rating",
        "score",
        "tag",
        "why_qualifies"
    ]

    existing_cols = [c for c in display_cols if c in pretty.columns]
    pretty = pretty[existing_cols]

    pretty = pretty.rename(columns={
        "driver_id": "Driver",
        "vehicle_type": "Vehicle",
        "pickup_distance": "Pickup Dist",
        "trip_distance": "Trip Dist",
        "wait_time": "Wait Time",
        "arrival_category": "Arrival",
        "price": "Price",
        "driver_min_accept_price": "Min Accept",
        "rating": "Rating",
        "score": "Score",
        "tag": "Tag",
        "why_qualifies": "Why Qualifies"
    })

    return pretty


def print_ride_summary(
    pickup_x,
    pickup_y,
    drop_x,
    drop_y,
    preference,
    vehicle_type,
    min_budget,
    max_budget,
    pickup_name=None,
    drop_name=None
):
    print("\n" + "=" * 52)
    print("🚕 RIDE REQUEST SUMMARY")
    print("=" * 52)

    if pickup_name and drop_name:
        print(f"Pickup Location : {pickup_name.title()} ({pickup_x}, {pickup_y})")
        print(f"Drop Location   : {drop_name.title()} ({drop_x}, {drop_y})")
    else:
        print(f"Pickup Location : ({pickup_x}, {pickup_y})")
        print(f"Drop Location   : ({drop_x}, {drop_y})")

    print(f"Preference      : {preference.title()}")
    print(f"Vehicle Type    : {vehicle_type.title()}")
    print(f"Budget Range    : ${min_budget:.2f} - ${max_budget:.2f}")
    print("=" * 52)


def print_best_choice(ranked, preference):
    """Print a short explanation of the top-ranked driver."""
    if ranked.empty:
        return

    best = ranked.iloc[0]

    if preference == "cheap":
        reason = "lowest price within your budget"
    elif preference == "fast":
        reason = "lowest estimated wait time"
    elif preference == "premium":
        reason = "highest driver rating"
    else:
        reason = "best overall score"

    print("\n⭐ BEST MATCH")
    print("-" * 52)
    print(f"Driver       : {best['driver_id']}")
    print(f"Vehicle      : {best['vehicle_type'].title()}")
    print(f"Reason       : Selected for {reason}")
    print(f"Price        : ${best['price']:.2f}")
    print(f"Wait Time    : {best['wait_time']:.1f} min ({best['arrival_category']})")
    print(f"Rating       : {best['rating']:.1f} / 5")
    print(f"Pickup Dist  : {best['pickup_distance']:.2f} km")
    print(f"Trip Dist    : {best['trip_distance']:.2f} km")
    print(f"Score        : {best['score']:.2f}")
    print(f"Tag          : {best['tag'] if best['tag'] else 'N/A'}")
    print("-" * 52)


def suggest_budget(offers, max_budget):
    """Suggest the next minimum budget if no driver is available."""
    if offers.empty:
        return None

    cheapest = offers["price"].min()
    if cheapest > max_budget:
        return round(cheapest, 2)
    return None


def plot_live_request_graph(ranked, pickup_x, pickup_y, drop_x, drop_y):
    """Plot a simple graph-like view of passenger and feasible drivers."""
    if ranked.empty:
        return

    plt.figure(figsize=(9, 7))

    # Plot pickup and drop
    plt.scatter(pickup_x, pickup_y, s=200, marker="o", label="Pickup")
    plt.scatter(drop_x, drop_y, s=200, marker="X", label="Drop")

    # Plot feasible drivers
    for _, row in ranked.iterrows():
        x = row["driver_x"]
        y = row["driver_y"]
        plt.scatter(x, y, s=120)
        plt.text(x + 0.5, y + 0.5, row["driver_id"], fontsize=9)

        # Draw line from driver to pickup
        plt.plot([x, pickup_x], [y, pickup_y], linestyle="--", linewidth=1)

    # Highlight best driver
    best = ranked.iloc[0]
    plt.scatter(best["driver_x"], best["driver_y"], s=260, marker="*", label=f"Best: {best['driver_id']}")

    plt.title("Live Ride Request Graph View")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    print("\n=== Ride Request Demo ===")

    trip_input = get_user_trip_input()
    if trip_input is None:
        return

    pickup_x, pickup_y, drop_x, drop_y, pickup_name, drop_name = trip_input

    preference = input("\nPreference (cheap / fast / premium / balanced): ").strip().lower()

    print("\nAvailable vehicle types:", ", ".join(VEHICLE_OPTIONS.keys()))
    vehicle_type = input("Enter vehicle type (economy / standard / premium): ").strip().lower()
    if vehicle_type not in VEHICLE_OPTIONS:
        print("Invalid vehicle type.")
        return

    min_budget = float(input("Enter the minimum price you expect to pay: "))
    max_budget = float(input("Enter the maximum price you want to pay: "))

    if min_budget > max_budget:
        print("Minimum budget cannot be greater than maximum budget.")
        return

    print_ride_summary(
        pickup_x, pickup_y, drop_x, drop_y,
        preference, vehicle_type, min_budget, max_budget,
        pickup_name, drop_name
    )

    drivers = load_drivers()
    offers = generate_offers(drivers, pickup_x, pickup_y, drop_x, drop_y, vehicle_type)

    if offers.empty:
        print("\nNo offers could be generated. Please check driver availability data.")
        return

    # Filter by passenger budget range
    budget_filtered = offers[
        (offers["price"] >= min_budget) &
        (offers["price"] <= max_budget)
    ].copy()

    # Driver acceptance rule: only show drivers who accept the offered fare
    accepted_offers = budget_filtered[
        budget_filtered["price"] >= budget_filtered["driver_min_accept_price"]
    ].copy()

    if accepted_offers.empty:
        print("\n❌ No drivers are available within your budget range and acceptance rules.")
        suggestion = suggest_budget(offers, max_budget)
        if suggestion is not None:
            print(f"💡 Suggestion: Increase your maximum budget to at least ${suggestion:.2f} to see the cheapest driver.")
        return

    ranked = rank_offers(accepted_offers, preference)
    ranked = add_recommendation_tags(ranked)

    ranked["why_qualifies"] = ranked.apply(
        lambda row: explain_qualification(row, min_budget, max_budget, preference),
        axis=1
    )

    print(f"\n✅ {len(ranked)} driver(s) found within your budget range.")

    print_best_choice(ranked, preference)

    print("\n📋 ALL AVAILABLE DRIVERS WITHIN YOUR BUDGET RANGE")
    print("-" * 52)
    formatted_all = format_display_table(ranked)
    print(formatted_all.to_string(index=False))

    print("\n🏆 TOP 3 RECOMMENDED DRIVERS")
    print("-" * 52)
    formatted_top3 = format_display_table(ranked.head(3))
    print(formatted_top3.to_string(index=False))

    # Graph plot
    plot_choice = input("\nDo you want to see the graph plot? (yes/no): ").strip().lower()
    if plot_choice in {"yes", "y"}:
        plot_live_request_graph(ranked, pickup_x, pickup_y, drop_x, drop_y)


if __name__ == "__main__":
    main()
    