"""Airline Flight Data Analysis - Streamlit dashboard with basic ML.

Based on the "Airline Flight Data Analysis" deck (BTS route data via vega-datasets).
Run:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).parent / "data"

st.set_page_config(page_title="Airline Flight Data Analysis", page_icon="✈️", layout="wide")


# --------------------------------------------------------------------------- data
def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


@st.cache_data
def load_data(min_flights: int = 1):
    routes = pd.read_csv(DATA / "flights-airport.csv")
    routes = routes[routes["count"] >= min_flights]
    airports = pd.read_csv(DATA / "airports.csv")
    ap = airports.set_index("iata")

    routes = routes.join(ap[["latitude", "longitude", "city"]].add_prefix("o_"), on="origin")
    routes = routes.join(ap[["latitude", "longitude", "city"]].add_prefix("d_"), on="destination")
    routes = routes.dropna(subset=["o_latitude", "d_latitude"]).copy()
    routes["distance"] = haversine(
        routes.o_latitude, routes.o_longitude, routes.d_latitude, routes.d_longitude
    )

    dep = routes.groupby("origin")["count"].sum().rename("departures")
    arr = routes.groupby("destination")["count"].sum().rename("arrivals")
    deg_out = routes.groupby("origin")["destination"].nunique().rename("dest_count")
    deg_in = routes.groupby("destination")["origin"].nunique().rename("origin_count")
    avg_dist = (
        routes.assign(w=routes["distance"] * routes["count"])
        .groupby("origin")
        .apply(lambda g: g["w"].sum() / g["count"].sum(), include_groups=False)
        .rename("avg_dist")
    )
    stats = pd.concat([dep, arr, deg_out, deg_in, avg_dist], axis=1).fillna(0)
    stats["total"] = stats["departures"] + stats["arrivals"]
    stats["connections"] = stats["dest_count"] + stats["origin_count"]
    stats = stats.join(ap[["name", "city", "state", "latitude", "longitude"]])
    stats = stats.reset_index(names="iata").sort_values("total", ascending=False)
    return routes, stats




# --------------------------------------------------------------------------- ML helpers
FEATURES = ["log_distance", "origin_out", "dest_in", "origin_deg", "dest_deg"]
FEATURE_LABELS = {
    "log_distance": "Distance (log miles)",
    "origin_out": "Origin other departures (log)",
    "dest_in": "Destination other arrivals (log)",
    "origin_deg": "Origin # destinations (log)",
    "dest_deg": "Destination # origins (log)",
}


@st.cache_data
def build_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features for route-volume prediction.
    Airport traffic features exclude the route itself to avoid target leakage."""
    out_tot = df.groupby("origin")["count"].transform("sum")
    in_tot = df.groupby("destination")["count"].transform("sum")
    out_deg = df.groupby("origin")["destination"].transform("nunique")
    in_deg = df.groupby("destination")["origin"].transform("nunique")
    f = pd.DataFrame(
        {
            "origin": df["origin"],
            "destination": df["destination"],
            "log_distance": np.log1p(df["distance"]),
            "origin_out": np.log1p(out_tot - df["count"]),
            "dest_in": np.log1p(in_tot - df["count"]),
            "origin_deg": np.log1p(out_deg - 1),
            "dest_deg": np.log1p(in_deg - 1),
            "y": np.log1p(df["count"]),
        }
    )
    return f


def make_model(name: str, seed: int):
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(random_state=seed)
    return LinearRegression()


@st.cache_resource
def train_route_model(name: str, test_size: float, seed: int, min_flights: int):
    feats = build_route_features(routes)
    X, y = feats[FEATURES], feats["y"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
    model = make_model(name, seed).fit(Xtr, ytr)
    pred = model.predict(Xte)
    metrics = {
        "R2": r2_score(yte, pred),
        "MAE (flights)": mean_absolute_error(np.expm1(yte), np.expm1(pred)),
        "Train rows": len(Xtr),
        "Test rows": len(Xte),
    }
    test_df = pd.DataFrame({"actual": np.expm1(yte), "predicted": np.expm1(pred)})
    test_df = test_df.join(feats[["origin", "destination"]])
    return model, metrics, test_df


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("✈️ Flight Analysis")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Busiest airports", "Busiest routes", "Network map", "ML: predict route volume",
     "ML: hub clustering", "Insights"],
)
min_flights = st.sidebar.slider(
    "Min flights per route", 1, 2000, 1,
    help="Drop low-volume routes to focus on the main network (the deck filtered to hubs).")
routes, stats = load_data(min_flights)
st.sidebar.caption(f"{len(routes):,} routes · {len(stats)} airports")
st.sidebar.caption("Data: Bureau of Transportation Statistics via vega-datasets")


# --------------------------------------------------------------------------- pages
if page == "Overview":
    st.title("US Domestic Aviation: Airline Flight Data Analysis")
    st.caption("Where America flies: hubs, routes and the shape of the network")

    top_route = routes.sort_values("count", ascending=False).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flights per year", f"{routes['count'].sum() / 1e6:.2f}M")
    c2.metric("Airports served", f"{len(stats)}")
    c3.metric("Distinct routes", f"{len(routes):,}")
    c4.metric(f"Busiest: {top_route.origin} → {top_route.destination}", f"{top_route['count']:,}")

    st.subheader("Objectives")
    st.markdown(
        "1. Which airports handle the most traffic?\n"
        "2. Which city pairs are the busiest routes?\n"
        "3. How concentrated is the network around hubs?"
    )

    st.subheader("Hub concentration")
    n = st.slider("Top N airports", 3, 30, 10)
    share = stats["total"].head(n).sum() / stats["total"].sum()
    st.write(f"The top **{n}** airports handle **{share:.1%}** of all departures + arrivals.")
    cum = (stats["total"].cumsum() / stats["total"].sum()).reset_index(drop=True)
    fig = px.line(x=np.arange(1, len(cum) + 1), y=cum * 100,
                  labels={"x": "Number of airports (ranked by traffic)", "y": "Cumulative % of traffic"})
    fig.add_vline(x=n, line_dash="dash")
    st.plotly_chart(fig, width="stretch")

    with st.expander("Preview raw data"):
        st.dataframe(routes[["origin", "destination", "count", "distance"]].head(100), width="stretch")

elif page == "Busiest airports":
    st.title("Busiest airports")
    st.caption("Departures + arrivals")
    n = st.slider("Number of airports", 5, 30, 15)
    top = stats.head(n).iloc[::-1]
    fig = px.bar(top, x="total", y="iata", orientation="h", hover_data=["name", "city", "departures", "arrivals"],
                 labels={"total": "Flights", "iata": "Airport"}, color="total",
                 color_continuous_scale="Blues")
    fig.update_layout(coloraxis_showscale=False, height=max(400, 22 * n))
    st.plotly_chart(fig, width="stretch")

    a, b = stats.iloc[0], stats.iloc[1]
    st.info(f"{a.iata} leads with about {a.total / 1000:,.0f}K flights, roughly "
            f"{(a.total / b.total - 1):.0%} more than second-place {b.iata}.")
    st.dataframe(
        stats[["iata", "name", "city", "state", "departures", "arrivals", "total", "connections"]].head(n),
        width="stretch", hide_index=True)

elif page == "Busiest routes":
    st.title("Busiest routes")
    c1, c2 = st.columns(2)
    n = c1.slider("Number of routes", 5, 30, 15)
    merge = c2.checkbox("Combine both directions (A↔B)", value=False)
    r = routes[["origin", "destination", "count", "distance"]].copy()
    if merge:
        r["a"] = r[["origin", "destination"]].min(axis=1)
        r["b"] = r[["origin", "destination"]].max(axis=1)
        r = r.groupby(["a", "b"], as_index=False).agg(count=("count", "sum"), distance=("distance", "first"))
        r["route"] = r["a"] + " ↔ " + r["b"]
    else:
        r["route"] = r["origin"] + " → " + r["destination"]
    top = r.sort_values("count", ascending=False).head(n)
    fig = px.bar(top.iloc[::-1], x="count", y="route", orientation="h", color="distance",
                 color_continuous_scale="Viridis",
                 labels={"count": "Annual flights", "distance": "Distance (mi)"})
    fig.update_layout(height=max(400, 24 * n))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Do short routes dominate?")
    fig2 = px.scatter(routes, x="distance", y="count", log_y=True, hover_data=["origin", "destination"],
                      labels={"distance": "Distance (miles)", "count": "Annual flights (log)"}, opacity=0.5)
    st.plotly_chart(fig2, width="stretch")

    st.subheader("Route symmetry")
    pair = routes.set_index(["origin", "destination"])["count"]
    rev = pair.reindex(pd.MultiIndex.from_arrays([routes["destination"], routes["origin"]])).values
    both = routes.assign(reverse=rev).dropna(subset=["reverse"])
    corr = np.corrcoef(both["count"], both["reverse"])[0, 1]
    st.write(f"Correlation between a route's volume and its reverse direction: **{corr:.3f}**")

elif page == "Network map":
    st.title("Network map")
    c1, c2 = st.columns(2)
    n_routes = c1.slider("Show top N routes", 10, 300, 80)
    n_air = c2.slider("Label top N airports", 5, 50, 20)
    top = routes.sort_values("count", ascending=False).head(n_routes)
    fig = go.Figure()
    mx = top["count"].max()
    for _, r in top.iterrows():
        fig.add_trace(go.Scattergeo(
            lon=[r.o_longitude, r.d_longitude], lat=[r.o_latitude, r.d_latitude], mode="lines",
            line=dict(width=0.5 + 3 * r["count"] / mx, color="rgba(30,100,200,0.4)"),
            hoverinfo="text", text=f"{r.origin} → {r.destination}: {r['count']:,}", showlegend=False))
    ap = stats.head(n_air)
    fig.add_trace(go.Scattergeo(
        lon=ap.longitude, lat=ap.latitude, mode="markers+text", text=ap.iata, textposition="top center",
        marker=dict(size=6 + 24 * ap.total / ap.total.max(), color="crimson", line=dict(width=1, color="white")),
        hovertext=ap.apply(lambda r: f"{r['name']}<br>{r.total:,.0f} flights", axis=1), hoverinfo="text",
        showlegend=False))
    fig.update_geos(scope="north america", projection_type="albers usa", showland=True,
                    landcolor="rgb(243,243,243)", subunitcolor="rgb(200,200,200)", showsubunits=True)
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

elif page == "ML: predict route volume":
    st.title("ML: predict route volume")
    st.markdown(
        "A supervised regression model predicts how many flights a route carries per year from "
        "its **distance** and the **size and connectivity of its two airports**. Airport features "
        "exclude the route itself so the model can't see the answer. "
        "(The deck's delay prediction needs delay data, which this dataset lacks.)")

    c1, c2, c3 = st.columns(3)
    algo = c1.selectbox("Algorithm", ["Random Forest", "Gradient Boosting", "Linear Regression"])
    test_size = c2.slider("Test set share", 0.1, 0.5, 0.2, 0.05)
    seed = c3.number_input("Random seed", 0, 9999, 42)

    model, metrics, test_df = train_route_model(algo, test_size, int(seed), min_flights)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² (log scale)", f"{metrics['R2']:.3f}")
    m2.metric("MAE (flights)", f"{metrics['MAE (flights)']:,.0f}")
    m3.metric("Train rows", f"{metrics['Train rows']:,}")
    m4.metric("Test rows", f"{metrics['Test rows']:,}")

    left, right = st.columns(2)
    with left:
        st.subheader("Actual vs predicted (test set)")
        fig = px.scatter(test_df, x="actual", y="predicted", log_x=True, log_y=True,
                         hover_data=["origin", "destination"], opacity=0.6)
        lim = [1, test_df[["actual", "predicted"]].max().max()]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color="gray"),
                                 name="perfect"))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Feature importance")
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        else:
            imp = np.abs(model.coef_) / np.abs(model.coef_).sum()
        imp_df = pd.DataFrame({"feature": [FEATURE_LABELS[f] for f in FEATURES], "importance": imp})
        st.plotly_chart(px.bar(imp_df.sort_values("importance"), x="importance", y="feature",
                               orientation="h"), width="stretch")

    st.subheader("Try a prediction")
    codes = stats["iata"].tolist()
    p1, p2 = st.columns(2)
    o = p1.selectbox("Origin", codes, index=0)
    d = p2.selectbox("Destination", codes, index=1)
    if o == d:
        st.warning("Choose two different airports.")
    else:
        so, sd = stats.set_index("iata").loc[o], stats.set_index("iata").loc[d]
        dist = haversine(so.latitude, so.longitude, sd.latitude, sd.longitude)
        row = pd.DataFrame([{
            "log_distance": np.log1p(dist),
            "origin_out": np.log1p(so.departures),
            "dest_in": np.log1p(sd.arrivals),
            "origin_deg": np.log1p(so.dest_count),
            "dest_deg": np.log1p(sd.origin_count),
        }])
        pred = float(np.expm1(model.predict(row[FEATURES])[0]))
        actual = routes[(routes.origin == o) & (routes.destination == d)]["count"]
        k1, k2, k3 = st.columns(3)
        k1.metric("Distance", f"{dist:,.0f} mi")
        k2.metric("Predicted annual flights", f"{max(pred, 0):,.0f}")
        k3.metric("Actual in data", f"{int(actual.iloc[0]):,}" if len(actual) else "No such route")

elif page == "ML: hub clustering":
    st.title("ML: hub clustering")
    st.markdown("Unsupervised **K-Means** groups airports by traffic, connectivity and typical trip length "
                "to separate major hubs from regional airports.")
    k = st.slider("Number of clusters", 2, 6, 4)
    feats = pd.DataFrame({
        "log_traffic": np.log1p(stats["total"]),
        "log_connections": np.log1p(stats["connections"]),
        "avg_dist": stats["avg_dist"],
    })
    X = StandardScaler().fit_transform(feats)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    s = stats.copy()
    s["cluster"] = km.labels_
    order = s.groupby("cluster")["total"].mean().sort_values(ascending=False).index
    names = {c: f"Tier {i + 1}" for i, c in enumerate(order)}
    s["tier"] = s["cluster"].map(names)

    summary = s.groupby("tier").agg(
        airports=("iata", "count"), avg_traffic=("total", "mean"),
        avg_connections=("connections", "mean"), avg_distance=("avg_dist", "mean"),
        examples=("iata", lambda x: ", ".join(x.head(5)))).round(0).reset_index()
    st.dataframe(summary, width="stretch", hide_index=True)

    tiers = sorted(s["tier"].unique())
    fig = px.scatter(s, x="connections", y="total", color="tier", log_x=True, log_y=True,
                     category_orders={"tier": tiers}, hover_data=["iata", "name", "city"],
                     labels={"connections": "Connections (log)", "total": "Flights (log)"})
    st.plotly_chart(fig, width="stretch")

    fig2 = px.scatter_geo(s, lat="latitude", lon="longitude", color="tier", size="total",
                          category_orders={"tier": tiers}, hover_name="name", scope="usa")
    fig2.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, width="stretch")

elif page == "Insights":
    st.title("Key insights")
    top4 = stats.head(4)["iata"].tolist()
    share = stats["total"].head(5).sum() / stats["total"].sum()
    st.markdown(
        f"**1. Hub-and-spoke.** A handful of hubs ({', '.join(top4)}) carry a disproportionate share of "
        f"traffic; the top 5 airports handle {share:.0%} of all departures + arrivals.\n\n"
        "**2. Symmetry.** Routes are near mirror images of each other (see the symmetry check on the "
        "*Busiest routes* page).\n\n"
        "**3. Corridors.** Top routes cluster in three regions: West Coast, Northeast and Southwest."
    )
    st.subheader("Next steps (from the deck)")
    st.markdown("- Add delay and cancellation data to link busy routes to punctuality\n"
                "- Airline-level comparison and seasonal trends\n"
                "- Predict delays with a machine learning model")
