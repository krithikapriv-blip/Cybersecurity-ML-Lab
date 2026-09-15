import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Create sample network traffic data
data = {
    "Traffic": [
        100, 110, 120, 130, 125, 140, 150,
        160, 170, 165, 180, 190, 200, 210,
        220, 215, 230, 240, 250, 245, 260,
        270, 280, 275, 290, 300, 310, 320,
        315, 330
    ]
}

df = pd.DataFrame(data)

# Decompose the time series
result = seasonal_decompose(
    df["Traffic"],
    model="additive",
    period=7
)

# Display components
print("Trend:")
print(result.trend.dropna())

print("\nSeasonal:")
print(result.seasonal.dropna())

print("\nResidual:")
print(result.resid.dropna())

# Plot decomposition
result.plot()
plt.suptitle("Network Traffic Time Series Decomposition")
plt.tight_layout()
plt.show()
