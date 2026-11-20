# Airline Flight Data Analysis (Streamlit + ML)

Interactive version of the "Airline Flight Data Analysis" deck: hubs, routes and network shape,
with two basic ML models.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pages
- **Overview**: KPIs and hub-concentration curve
- **Busiest airports / routes**: ranked charts, distance vs volume, route symmetry
- **Network map**: top routes and hubs on a US map
- **ML: predict route volume**: Random Forest / Gradient Boosting / Linear Regression, metrics, feature importance, what-if predictor
- **ML: hub clustering**: K-Means tiers of airports
- **Insights**: the deck's findings and next steps

## Data
`data/flights-airport.csv` (origin, destination, count) and `data/airports.csv` from
[vega-datasets](https://github.com/vega/vega-datasets) (BTS source).
