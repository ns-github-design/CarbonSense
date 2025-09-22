# Sustainability Tracker (Hackathon Starter)

A tiny Streamlit app where users log electricity (kWh) and commute (km) usage. 
The app estimates CO₂ emissions and shows simple savings/offset suggestions.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` — main Streamlit app
- `factors.json` — editable emission factors (kg CO₂ per unit)
- `requirements.txt` — Python deps
- `data.db` — SQLite database (auto-created)

> NOTE: Emission factors are placeholders. Update `factors.json` for your region if you have better values.
