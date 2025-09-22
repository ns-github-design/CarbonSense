
import streamlit as st
import pandas as pd
import sqlite3, json
from datetime import date

st.set_page_config(page_title="Carbon Sense", page_icon="🌱", layout="centered")

# --- Load & save factors ---
def load_factors():
    try:
        with open("factors.json", "r") as f:
            return json.load(f)
    except Exception:
        return {
            "electricity_kg_per_kwh": 0.5,
            "commute_kg_per_km": {
                "car_petrol": 0.192, "car_diesel": 0.171, "motorbike": 0.103,
                "bus": 0.105, "metro_train": 0.041, "walking": 0.0, "cycling": 0.0, "ev_car": 0.053
            },
            "baseline_commute_mode": "car_petrol"
        }

def save_factors(factors):
    with open("factors.json", "w") as f:
        json.dump(factors, f, indent=2)

factors = load_factors()

# --- DB ---
@st.cache_resource
def get_conn():
    conn = sqlite3.connect("data.db", check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entries(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            d TEXT NOT NULL,
            category TEXT NOT NULL,   -- 'electricity' or 'commute'
            amount REAL NOT NULL,     -- kWh for electricity, km for commute
            unit TEXT NOT NULL,
            mode TEXT,                -- commute mode (car, bus, etc.); NULL for electricity
            notes TEXT
        )
    """)
    return conn

conn = get_conn()

def add_entry(d, category, amount, unit, mode=None, notes=""):
    conn.execute(
        "INSERT INTO entries(d, category, amount, unit, mode, notes) VALUES(?,?,?,?,?,?)",
        (d, category, amount, unit, mode, notes),
    )
    conn.commit()

def load_entries():
    df = pd.read_sql_query("SELECT * FROM entries ORDER BY d DESC, id DESC", conn)
    if not df.empty:
        # compute emissions
        e_factor = factors.get("electricity_kg_per_kwh", 0.5)
        commute_f = factors.get("commute_kg_per_km", {})
        def _calc(row):
            if row["category"] == "electricity":
                return row["amount"] * e_factor
            else:
                return row["amount"] * float(commute_f.get(row["mode"], 0))
        df["kg_co2"] = df.apply(_calc, axis=1)
    return df

# --- UI ---
st.title("🌱 Carbon Sense")
st.caption("Log electricity and commute usage. Get CO₂ estimates and simple savings.")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "➕ Log Usage",
    "📊 Dashboard",
    "⚙️ Factors & Savings",
    "🎖 Badges",
    "💡 AI Tips",
    "📥 Data Import",
    "🎯 Goals"
])


with tab1:
    st.subheader("Add a new entry")
    c = st.radio("Category", ["Electricity (kWh)", "Commute (km)"], horizontal=True)
    d = st.date_input("Date", value=date.today())
    notes = st.text_input("Notes (optional)", placeholder="e.g., Work-from-home day, metro ride, AC usage")

    if c.startswith("Electricity"):
        kwh = st.number_input("Electricity used (kWh)", min_value=0.0, step=0.1)
        if st.button("Add electricity entry"):
            if kwh > 0:
                add_entry(str(d), "electricity", kwh, "kWh", None, notes)
                st.success("Added electricity entry.")
    else:
        mode = st.selectbox("Commute mode", list(factors["commute_kg_per_km"].keys()), index=0, format_func=lambda x: x.replace("_"," ").title())
        km = st.number_input("Distance (km)", min_value=0.0, step=0.1)
        if st.button("Add commute entry"):
            if km > 0:
                add_entry(str(d), "commute", km, "km", mode, notes)
                st.success("Added commute entry.")

    st.divider()
    st.subheader("Recent entries")
    df = load_entries()
    if df.empty:
        st.info("No entries yet. Add your first one above.")
    else:
        st.dataframe(df)

with tab2:
    st.subheader("Your CO₂ at a glance")
    df = load_entries()
    if df.empty:
        st.info("No data yet.")
    else:
        total = df["kg_co2"].sum()
        elec = df[df["category"]=="electricity"]["kg_co2"].sum()
        commute = df[df["category"]=="commute"]["kg_co2"].sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total CO₂ (kg)", f"{total:.2f}")
        col2.metric("Electricity (kg)", f"{elec:.2f}")
        col3.metric("Commute (kg)", f"{commute:.2f}")

        # category breakdown
        cat_breakdown = df.groupby("category", as_index=False)["kg_co2"].sum()
        st.bar_chart(cat_breakdown.set_index("category"))

        # --- Trends over time ---
        df["d"] = pd.to_datetime(df["d"])

        # Weekly totals
        weekly = df.resample("W-MON", on="d")["kg_co2"].sum().reset_index()
        st.subheader("📅 Weekly CO₂ Trend")
        st.line_chart(weekly.set_index("d"))

        # Monthly totals
        monthly = df.resample("M", on="d")["kg_co2"].sum().reset_index()
        st.subheader("📅 Monthly CO₂ Trend")
        st.line_chart(monthly.set_index("d"))

        # Compare last 2 weeks
        if len(weekly) >= 2:
            last = weekly.iloc[-1]["kg_co2"]
            prev = weekly.iloc[-2]["kg_co2"]
            change = ((last - prev) / prev * 100) if prev > 0 else 0
            st.metric("Change vs Last Week", f"{change:.1f}%")

            if change < 0:
                st.success("👏 Great job! Your CO₂ went down compared to last week.")
            elif change > 0:
                st.warning("⚠️ Your CO₂ went up this week. Try more public transport or reduce electricity use.")
            else:
                st.info("No change compared to last week.")


        # top commute modes by CO2
        dfm = df[df["category"]=="commute"]
        if not dfm.empty:
            modes = dfm.groupby("mode", as_index=False)["kg_co2"].sum().sort_values("kg_co2", ascending=False)
            st.caption("Commute CO₂ by mode")
            st.bar_chart(modes.set_index("mode"))

with tab3:
    st.subheader("Emission factors")
    st.caption("Tune these numbers for your region. Values are kg CO₂ per unit.")

    e_val = st.number_input("Electricity (kg CO₂ per kWh)", value=float(factors["electricity_kg_per_kwh"]), step=0.01, format="%.3f")
    st.write("Commute (kg CO₂ per km):")
    updated_commute = {}
    for k, v in factors["commute_kg_per_km"].items():
        updated_commute[k] = st.number_input(k.replace("_"," ").title(), value=float(v), step=0.001, format="%.3f")

    baseline = st.selectbox("Baseline mode for savings (what you'd otherwise use)", list(updated_commute.keys()), index=list(updated_commute.keys()).index(factors.get("baseline_commute_mode","car_petrol")))

    if st.button("Save factors"):
        factors["electricity_kg_per_kwh"] = float(e_val)
        factors["commute_kg_per_km"] = {k: float(v) for k, v in updated_commute.items()}
        factors["baseline_commute_mode"] = baseline
        save_factors(factors)
        st.success("Saved factors.")

    st.divider()
    st.subheader("Estimated savings vs baseline")
    df = load_entries()
    if df.empty:
        st.info("Add commute entries to see savings.")
    else:
        commute_f = factors["commute_kg_per_km"]
        base = float(commute_f.get(baseline, 0))
        def _s(row):
            if row["category"]=="commute":
                actual = float(commute_f.get(row["mode"], 0))
                return max(0.0, (base - actual) * row["amount"])
            return 0.0
        df["savings_kg"] = df.apply(_s, axis=1)
        total_savings = df["savings_kg"].sum()
        st.metric("CO₂ avoided (kg)", f"{total_savings:.2f}")
        st.caption("Savings estimate assumes you'd otherwise use the baseline mode above.")

        # Offsets suggestion (very rough, illustrative)
        if total_savings > 0:
            # Illustrative: 1 mature tree ~ 21 kg CO₂/year
            trees = total_savings / 21.0
            st.write(f"🌳 Roughly equivalent to **{trees:.1f} trees** of annual sequestration.")
        st.write("💡 Tips: take public transport for short trips, combine errands, switch to LED lighting, and monitor appliance usage.")

with tab4:
    st.subheader("🎖 Your Badges")

    df = load_entries()
    if df.empty:
        st.info("No data yet – start logging to unlock badges 🌱")
    else:
        # --- Collect stats ---
        df["d"] = pd.to_datetime(df["d"])
        commute_df = df[df["category"] == "commute"]

        total_savings = 0.0
        if "savings_kg" in df.columns:
            total_savings = df["savings_kg"].sum()

        # Non-car commute km
        non_car_modes = ["bus", "metro_train", "walking", "cycling", "ev_car", "motorbike"]
        non_car_km = commute_df[commute_df["mode"].isin(non_car_modes)]["amount"].sum()

        # Days logged
        days_logged = df["d"].nunique()

        stats = {
            "savings": total_savings,
            "non_car_km": non_car_km,
            "days_logged": days_logged,
        }

        # --- Badge rules ---
        BADGES = [
            {"name": "🌱 First 10 kg saved!", "condition": lambda s: s["savings"] >= 10},
            {"name": "🌱 25 kg CO₂ Hero!", "condition": lambda s: s["savings"] >= 25},
            {"name": "🚲 50 km without car", "condition": lambda s: s["non_car_km"] >= 50},
            {"name": "⚡ 7-Day Streak", "condition": lambda s: s["days_logged"] >= 7},
        ]

        # --- Check earned badges ---
        earned = [b["name"] for b in BADGES if b["condition"](stats)]

        if earned:
            for b in earned:
                st.success(f"🏆 {b}")
        else:
            st.info("No badges yet – keep going!")

        st.divider()

        # --- Progress bars for next milestones ---
        st.subheader("📊 Progress to Next Goals")
        st.progress(min(stats["savings"] / 25, 1.0))
        st.caption(f"{stats['savings']:.1f} / 25 kg CO₂ saved")

        st.progress(min(stats["non_car_km"] / 50, 1.0))
        st.caption(f"{stats['non_car_km']:.1f} / 50 km without car")

        st.progress(min(stats["days_logged"] / 7, 1.0))
        st.caption(f"{stats['days_logged']} / 7 days logged")

import os
from openai import OpenAI

with tab5:
    st.subheader("💡 AI-Powered Sustainability Tips")

    df = load_entries()
    if df.empty:
        st.info("No data yet – log some usage to get personalized tips 🌱")
    else:
        # --- Summarize recent data (last 7 days) ---
        df["d"] = pd.to_datetime(df["d"])
        last_week = df[df["d"] >= pd.Timestamp.today() - pd.Timedelta(days=7)]

        total = last_week["kg_co2"].sum()
        elec_kwh = last_week[last_week["category"]=="electricity"]["amount"].sum()
        commute_km = last_week[last_week["category"]=="commute"]["amount"].sum()
        car_km = last_week[(last_week["category"]=="commute") & (last_week["mode"].str.contains("car"))]["amount"].sum()
        bus_km = last_week[(last_week["category"]=="commute") & (last_week["mode"]=="bus")]["amount"].sum()
        metro_km = last_week[(last_week["category"]=="commute") & (last_week["mode"]=="metro_train")]["amount"].sum()

        summary = f"""
        In the last 7 days:
        - Total CO₂: {total:.1f} kg
        - Electricity used: {elec_kwh:.1f} kWh
        - Commute distance: {commute_km:.1f} km (Car: {car_km:.1f}, Bus: {bus_km:.1f}, Metro: {metro_km:.1f})
        """

        st.write("📊 Usage summary for AI:")
        st.code(summary.strip())

        # --- Button to get AI tips ---
        if st.button("✨ Get AI Suggestions"):
            with st.spinner("Thinking..."):
                try:
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                    prompt = f"""
                    You are a sustainability coach.
                    Based on this data, suggest 2-3 practical, personalized tips to reduce CO₂ emissions:
                    {summary}
                    """

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a friendly sustainability assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=200,
                        temperature=0.7,
                    )

                    tips = response.choices[0].message.content
                    st.success("Here are your personalized tips:")
                    st.write(tips)

                except Exception as e:
                    st.error(f"Error fetching AI suggestions: {e}")
                    st.info("👉 Make sure you set your OPENAI_API_KEY in the environment before running the app.")

import pandas as pd
import os
import googlemaps
from datetime import date

with tab6:
    st.subheader("📥 Import Data Automatically")

    # --- Section A: Upload CSV for Electricity Bills ---
    st.markdown("### ⚡ Import Electricity Data from CSV")
    st.caption("Upload a CSV with columns: `Date` and `kWh`")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            bills = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(bills.head())

            for _, row in bills.iterrows():
                try:
                    add_entry(str(row["Date"]), "electricity", float(row["kWh"]), "kWh", None, "Imported from CSV")
                except Exception as e:
                    st.warning(f"Skipped row due to error: {e}")

            st.success("Electricity data imported ✅")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    st.divider()

    # --- Section B: Auto-calc commute distance with Google Maps ---
    st.markdown("### 🚗 Import Commute Distance (Google Maps API)")
    st.caption("Enter start & destination to auto-calc commute distance and add entry.")

    origin = st.text_input("Start location", placeholder="e.g., Bangalore MG Road")
    destination = st.text_input("Destination", placeholder="e.g., Bangalore Whitefield")
    mode = st.selectbox("Mode", ["driving", "transit", "walking", "bicycling"])

    if st.button("Calculate & Import Commute"):
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not api_key:
            st.error("Missing GOOGLE_MAPS_API_KEY. Set it as an environment variable first.")
        else:
            try:
                gmaps = googlemaps.Client(key=api_key)
                result = gmaps.distance_matrix(origin, destination, mode=mode)
                distance_m = result["rows"][0]["elements"][0]["distance"]["value"]
                distance_km = distance_m / 1000.0
                st.success(f"Distance: {distance_km:.1f} km")

                add_entry(str(date.today()), "commute", distance_km, "km", mode, f"{origin} → {destination}")
                st.info("Commute entry added automatically ✅")
            except Exception as e:
                st.error(f"Error fetching distance: {e}")

import calendar
import pandas as pd

with tab7:
    st.subheader("🎯 Monthly Goal & Forecast")

    df = load_entries()
    if df.empty:
        st.info("No data yet – log usage to set and track goals 🌱")
    else:
        # --- Input goal ---
        goal = st.number_input("Set your monthly CO₂ reduction goal (kg)", min_value=1, step=1, value=50)

        # --- Filter this month's data ---
        df["d"] = pd.to_datetime(df["d"])
        today = pd.Timestamp.today()
        this_month = df[df["d"].dt.month == today.month]

        this_month_total = this_month["kg_co2"].sum()

        # --- Daily average and forecast ---
        days_so_far = today.day
        avg_per_day = this_month_total / days_so_far if days_so_far > 0 else 0
        days_in_month = calendar.monthrange(today.year, today.month)[1]
        forecast_total = avg_per_day * days_in_month

        # --- Compare to goal ---
        progress = min(this_month_total / goal, 1.0)
        forecast_pct = (forecast_total / goal) * 100 if goal > 0 else 0

        # --- Show metrics ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Goal (kg)", f"{goal}")
        col2.metric("So far (kg)", f"{this_month_total:.1f}")
        col3.metric("Forecast (kg)", f"{forecast_total:.1f}")

        st.progress(progress)
        st.caption(f"You’re at {this_month_total:.1f} kg vs {goal} kg goal")

        if forecast_total <= goal:
            st.success(f"👏 At current pace, you’re on track to meet {forecast_pct:.0f}% of your goal.")
        else:
            st.warning(f"⚠️ At current pace, you may exceed your goal ({forecast_pct:.0f}%).")
