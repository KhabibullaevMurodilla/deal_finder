"""
Anywhere Deal Finder
---------------------
Pulls live flight prices from the Travelpayouts Data API, keeps a running
history of prices per route, and uses an anomaly-detection model
(Isolation Forest) to flag which current prices are genuinely unusual
"great deals" versus just normal day-to-day price variation.

HOW TO USE:
1. Get your Data API token: log into Travelpayouts -> Tools -> API
   (this is different from your affiliate marker/link - it's a separate
   token specifically for pulling price data)
2. Paste your token into API_TOKEN below
3. Run this script (locally, or in a free Google Colab notebook)
4. It saves:
   - price_history.csv   (grows every time you run it - this is what
                           lets the model learn what "normal" looks like)
   - todays_deals.json    (the ranked list of flagged deals)
   - telegram_post.txt    (a ready-to-paste post for your channel)

Run this daily (or every few days) - the more history it builds up,
the smarter the deal detection gets over time.
"""

import requests
import pandas as pd
import json
import os
from datetime import datetime, timezone
from sklearn.ensemble import IsolationForest

# ============================================================
# CONFIG - edit these
# ============================================================

# SECURITY: the token is read from an environment variable, not
# hardcoded here. This means it's never visible in the code itself,
# so it's safe to put this script in a public GitHub repo.
#
# To run LOCALLY on your own computer, set it before running:
#   Mac/Linux:  export TRAVELPAYOUTS_API_TOKEN="your_token_here"
#   Windows:    set TRAVELPAYOUTS_API_TOKEN=your_token_here
#
# To run via GITHUB ACTIONS, store it as a Repository Secret named
# TRAVELPAYOUTS_API_TOKEN (Settings -> Secrets and variables -> Actions)
# - the workflow file will pass it in automatically.
API_TOKEN = os.environ.get("TRAVELPAYOUTS_API_TOKEN", "")

# Routes to track: (origin IATA, destination IATA, human-readable name)
ROUTES = [
    ("TAS", "IST", "Tashkent -> Istanbul"),
    ("TAS", "DXB", "Tashkent -> Dubai"),
    ("TAS", "ICN", "Tashkent -> Seoul"),
    ("TAS", "BKK", "Tashkent -> Bangkok"),
    ("SKD", "MOW", "Samarkand -> Moscow"),
    ("TAS", "BCN", "Tashkent -> Barcelona"),
    ("TAS", "VCE", "Tashkent -> Venice"),
    ("TAS", "LON", "Tashkent -> London"),
]

CURRENCY = "usd"
HISTORY_FILE = "price_history.csv"
DEALS_OUTPUT = "todays_deals.json"
TELEGRAM_OUTPUT = "telegram_post.txt"

# ============================================================
# STEP 1 - Fetch current cheapest price for each route
# ============================================================

def fetch_price(origin, destination, token):
    """Calls the modern /aviasales/v3/prices_for_dates endpoint for a single route
    (replaces the legacy /v1/prices/cheap endpoint, per Travelpayouts docs)."""
    url = "https://api.travelpayouts.com/aviasales/v3/prices_for_dates"
    params = {
        "origin": origin,
        "destination": destination,
        "currency": CURRENCY,
        "sorting": "price",
        "direct": "false",
        "one_way": "true",
        "limit": 1,
    }
    headers = {"x-access-token": token}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        print(f"  [!] Request failed for {origin}->{destination}: {e}")
        return None

    if not payload.get("success"):
        print(f"  [!] API error for {origin}->{destination}: {payload.get('error')}")
        return None

    data = payload.get("data", [])
    if not data:
        print(f"  [!] No price data returned for {origin}->{destination}")
        return None

    return data[0].get("price")


def fetch_special_offers(origin, token):
    """Calls Travelpayouts' own 'abnormally low price' detector - a second,
    officially-flagged source of genuine deals, separate from our own model."""
    url = "https://api.travelpayouts.com/aviasales/v3/get_special_offers"
    params = {"origin": origin, "currency": CURRENCY, "locale": "en"}
    headers = {"x-access-token": token}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        print(f"  [!] Special offers request failed for {origin}: {e}")
        return []

    if not payload.get("success"):
        return []

    return payload.get("data", [])


def collect_current_prices():
    print("Fetching current prices for all tracked routes...")
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for origin, destination, label in ROUTES:
        price = fetch_price(origin, destination, API_TOKEN)
        if price is not None:
            print(f"  {label}: ${price}")
            rows.append({
                "timestamp": now,
                "origin": origin,
                "destination": destination,
                "route_label": label,
                "price": price,
            })
    return pd.DataFrame(rows)


def collect_special_offers():
    """Pulls Travelpayouts' own flagged 'abnormally low price' deals for each
    unique origin city we track - a second, independently-verified deal source."""
    print("\nChecking Travelpayouts' own special-offers detector...")
    origins = sorted(set(o for o, _, _ in ROUTES))
    offers = []
    for origin in origins:
        results = fetch_special_offers(origin, API_TOKEN)
        for r in results:
            offers.append({
                "route_label": f"{r.get('origin_name', origin)} -> {r.get('destination_name', r.get('destination'))}",
                "price": r.get("price"),
                "title": r.get("title"),
                "airline": r.get("airline_title"),
            })
        if results:
            print(f"  {origin}: {len(results)} special offer(s) found")
    return offers


# ============================================================
# STEP 2 - Append to history so the model has data to learn from
# ============================================================

def update_history(new_rows: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(HISTORY_FILE):
        history = pd.read_csv(HISTORY_FILE)
        history = pd.concat([history, new_rows], ignore_index=True)
    else:
        history = new_rows
    history.to_csv(HISTORY_FILE, index=False)
    return history


# ============================================================
# STEP 3 - Flag genuinely unusual ("great") deals
# ============================================================

def score_deals(history: pd.DataFrame) -> pd.DataFrame:
    """
    For each route, use an Isolation Forest to check whether today's
    price is an outlier compared to that route's own price history.
    Falls back to a simple 'below average' rule if there isn't enough
    history yet for a route (Isolation Forest needs a handful of points).
    """
    results = []
    for route_label, group in history.groupby("route_label"):
        group = group.sort_values("timestamp")
        prices = group[["price"]].values
        latest_price = group.iloc[-1]["price"]
        avg_price = group["price"].mean()

        if len(group) >= 5:
            model = IsolationForest(contamination=0.2, random_state=42)
            model.fit(prices)
            # score_samples: lower = more anomalous. We only care about
            # anomalies that are CHEAP, not expensive, so we check direction.
            latest_score = model.score_samples([[latest_price]])[0]
            is_anomaly = model.predict([[latest_price]])[0] == -1
            is_deal = is_anomaly and latest_price < avg_price
            method = "isolation_forest"
        else:
            # Not enough history yet - simple threshold instead
            is_deal = latest_price < avg_price * 0.85
            latest_score = None
            method = "simple_threshold_not_enough_history"

        pct_below_avg = round((1 - latest_price / avg_price) * 100, 1) if avg_price else 0

        results.append({
            "route_label": route_label,
            "latest_price": latest_price,
            "average_price": round(avg_price, 2),
            "pct_below_average": pct_below_avg,
            "is_deal": bool(is_deal),
            "detection_method": method,
            "data_points_so_far": len(group),
        })

    df = pd.DataFrame(results)
    return df.sort_values("pct_below_average", ascending=False)


# ============================================================
# STEP 4 - Output results
# ============================================================

def save_outputs(scored: pd.DataFrame, special_offers: list):
    scored.to_json(DEALS_OUTPUT, orient="records", indent=2)

    lines = ["FLEXIBLE DEALS - TODAY'S PICKS\n"]

    deals = scored[scored["is_deal"]]
    if not deals.empty:
        lines.append("From our own price tracking:")
        for _, row in deals.iterrows():
            lines.append(
                f"  {row['route_label']}: ${row['latest_price']} "
                f"({row['pct_below_average']}% below its usual average)"
            )
        lines.append("")

    if special_offers:
        lines.append("Flagged by Travelpayouts as abnormally low right now:")
        for offer in special_offers[:10]:
            lines.append(f"  {offer['route_label']}: ${offer['price']} ({offer['airline']})")
        lines.append("")

    if deals.empty and not special_offers:
        lines.append("No standout deals right now - prices look normal across tracked routes.")

    lines.append("Book via: https://aviasales.tpm.lv/3zOHKKXL")

    with open(TELEGRAM_OUTPUT, "w") as f:
        f.write("\n".join(lines))

    print("\n" + "=" * 50)
    print("\n".join(lines))
    print("=" * 50)
    print(f"\nSaved: {DEALS_OUTPUT}, {TELEGRAM_OUTPUT}, {HISTORY_FILE}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if not API_TOKEN:
        print("!! TRAVELPAYOUTS_API_TOKEN environment variable is not set.")
        print("   See the SECURITY note near the top of this file for how to set it.")
    else:
        current = collect_current_prices()
        offers = collect_special_offers()
        if current.empty and not offers:
            print("No prices were fetched - check your API token and route codes.")
        else:
            if not current.empty:
                full_history = update_history(current)
                scored = score_deals(full_history)
            else:
                scored = pd.DataFrame(columns=["route_label", "latest_price", "average_price",
                                                 "pct_below_average", "is_deal",
                                                 "detection_method", "data_points_so_far"])
            save_outputs(scored, offers)
