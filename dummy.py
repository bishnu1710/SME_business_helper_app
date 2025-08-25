import pandas as pd
import numpy as np

# Generate 120 months starting Jan 2015
dates = pd.date_range("2015-01-01", periods=120, freq="MS")

# Sales = base + upward trend + seasonal effect + random noise
sales = (
    1000
    + np.arange(120) * 50  # steady upward trend
    + 500 * np.sin(np.linspace(0, 20*np.pi, 120))  # yearly seasonality
    + np.random.randint(-200, 200, 120)  # random noise
)

df = pd.DataFrame({"Date": dates, "Sales": sales.astype(int)})
df.to_csv("dummy_sales_10yrs.csv", index=False)

print("✅ dummy_sales_10yrs.csv created with", len(df), "rows")
print(df.head())
