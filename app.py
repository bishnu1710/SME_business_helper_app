# business_helper_app.py
import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import holidays
import yfinance as yf

st.set_page_config(page_title="Business Helper App", layout="wide")

st.title("📊 Business Helper App – Context-Aware Sales Forecasting Tool")
st.write("Upload your sales data and let the app automatically enrich it with holidays and market index trends for realistic forecasting.")

# --- Upload Sales Data ---
uploaded_file = st.file_uploader("Upload Sales Data (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Read data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📂 Raw Sales Data")
    st.write(df.head())

    # Expecting columns: Date, Sales
    df.columns = [col.strip().lower() for col in df.columns]
    if "date" not in df.columns or "sales" not in df.columns:
        st.error("Dataset must have columns: Date, Sales")
    else:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        st.line_chart(df.set_index("date")["sales"])

        # --- Country selection for holiday calendar ---
        country = st.selectbox("Select Country for Holidays", ["India", "US", "UK"])

        # --- Fetch holidays dynamically ---
        years = df["date"].dt.year.unique()
        if country == "India":
            hol = holidays.India(years=years)
        elif country == "US":
            hol = holidays.US(years=years)
        else:
            hol = holidays.UnitedKingdom(years=years)
        # hol_df = pd.DataFrame(list(hol.items()), columns=["date", "holiday"])
        hol_df = pd.DataFrame(list(hol.items()), columns=["date", "holiday"])
        hol_df["date"] = pd.to_datetime(hol_df["date"])   # ensure datetime
        st.subheader("📅 Holidays Fetched")
        st.write(hol_df.head())

     
        @st.cache_data
        def fetch_market_data(ticker, start, end):
            return yf.download(
                ticker,
                start=start,
                end=end,
                interval="1mo",   # monthly data = faster + smaller
                progress=False
            )

      
        ticker = "^NSEI" if country == "India" else ("^GSPC" if country == "US" else "^FTSE")
        start_date = max(df["date"].min(), pd.Timestamp.today() - pd.DateOffset(years=5))

        st.info(f"📡 Fetching {ticker} market index as proxy for inflation...")   # moved here

        infl_df = fetch_market_data(ticker, start=start_date, end=pd.Timestamp.today())

        # --- Fix MultiIndex columns if present ---
        if isinstance(infl_df.columns, pd.MultiIndex):
            infl_df.columns = ["_".join([c for c in col if c]) for col in infl_df.columns]

        infl_df = infl_df.reset_index()

        # --- Dynamically detect a 'Close' column ---
        close_cols = [c for c in infl_df.columns if "Close" in c]

        if close_cols:
            chosen_col = close_cols[0]
            infl_df = infl_df.rename(columns={chosen_col: "inflation", "Date": "date"})
        else:
            st.error("❌ Could not fetch market data — no Close/Adj Close column found.")
            infl_df = pd.DataFrame(columns=["date", "inflation"])
        st.subheader("📈 Market Data Fetched")
        st.write(infl_df.head())

        # --- Merge contextual features ---
        df = pd.merge(df, infl_df, on="date", how="left")
        df = pd.merge(df, hol_df[["date"]].assign(is_holiday=1), on="date", how="left")
        df["is_holiday"] = df["is_holiday"].fillna(0)

        # Forecast horizon
        periods = st.slider("Months to Forecast", 1, 24, 6)

        # Model choice
        model_choice = st.radio("Choose Algorithm", ["Prophet", "XGBoost"])

        if st.button("Generate Forecast"):
            if model_choice == "Prophet":
                prophet_df = df.rename(columns={"date": "ds", "sales": "y"})
                holidays_prophet = hol_df.rename(columns={"date": "ds", "holiday": "holiday"})

                model = Prophet(yearly_seasonality=True, holidays=holidays_prophet)
                model.fit(prophet_df)

                future = model.make_future_dataframe(periods=periods, freq="M")
                forecast = model.predict(future)

                st.subheader("📈 Forecast Plot with Holidays")
                fig = model.plot(forecast)
                st.pyplot(fig)

                st.subheader("🔮 Forecasted Values")
                st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods))

                # --- Cross-validation for overfitting check ---
                st.subheader("📊 Model Validation (Prophet CV)")
                try:
                    df_cv = cross_validation(
                        model, 
                        initial='730 days',   # first 2 years for training
                        period='180 days',    # step size
                        horizon='365 days'    # forecast 1 year into future
                    )
                    df_p = performance_metrics(df_cv)
                    st.write(df_p[["horizon", "mae", "mape", "rmse"]].head())
                except Exception as e:
                    st.warning(f"Cross-validation failed: {e}")

            elif model_choice == "XGBoost":
                df["month"] = df["date"].dt.month
                df["year"] = df["date"].dt.year

                features = ["month", "year", "inflation", "is_holiday"]
                X = df[features].fillna(method="ffill")
                y = df["sales"]

                # --- Time Series Split (instead of random train_test_split) ---
                st.subheader("📊 Model Validation (XGBoost CV)")
                tscv = TimeSeriesSplit(n_splits=5)

                fold_results = []
                for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    model = XGBRegressor(n_estimators=200, learning_rate=0.1)
                    model.fit(X_train, y_train)

                    preds = model.predict(X_test)
                    mae = mean_absolute_error(y_test, preds)
                    rmse = mean_squared_error(y_test, preds, squared=False)

                    fold_results.append({"fold": fold+1, "mae": mae, "rmse": rmse})

                st.write(pd.DataFrame(fold_results))

                # --- Train on full data for final forecast ---
                model = XGBRegressor(n_estimators=200, learning_rate=0.1)
                model.fit(X, y)

                future_months = pd.date_range(df["date"].max(), periods=periods+1, freq="M")[1:]
                future_df = pd.DataFrame({
                    "month": future_months.month,
                    "year": future_months.year,
                    "inflation": df["inflation"].ffill().iloc[-1],
                    "is_holiday": 0
                })

                forecast = model.predict(future_df)
                result = pd.DataFrame({"date": future_months, "forecast_sales": forecast})

                st.subheader("📈 Forecast Plot with Contextual Features")
                fig, ax = plt.subplots()
                ax.plot(df["date"], df["sales"], label="Historical")
                ax.plot(result["date"], result["forecast_sales"], label="Forecast", linestyle="--")
                ax.legend()
                st.pyplot(fig)

                st.subheader("🔮 Forecasted Values")
                st.write(result)

            