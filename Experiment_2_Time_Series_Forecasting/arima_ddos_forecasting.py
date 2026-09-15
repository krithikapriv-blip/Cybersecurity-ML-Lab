import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Sample historical network traffic data
traffic = [
    100, 105, 110, 115, 120,
    118, 125, 130, 135, 140,
    138, 145, 150, 155, 160,
    158, 165, 170, 175, 180,
    178, 185, 190, 195, 200,
    198, 205, 210, 215, 220,
    218, 225, 230, 235, 240,
    300, 320, 340, 360, 380
]

# Create DataFrame
df = pd.DataFrame({
    "Traffic": traffic
})

# Create ARIMA model
model = ARIMA(df["Traffic"], order=(1, 1, 1))

# Train the model
model_fit = model.fit()

# Forecast the next 5 values
forecast = model_fit.forecast(steps=5)

print("Forecasted Network Traffic:")
print(forecast)

# Plot historical traffic
plt.plot(
    df["Traffic"],
    label="Historical Traffic"
)

# Plot forecast
forecast_index = range(
    len(df),
    len(df) + 5
)

plt.plot(
    forecast_index,
    forecast,
    marker="o",
    label="Forecast"
)

plt.xlabel("Time")
plt.ylabel("Network Traffic")
plt.title("ARIMA Network Traffic Forecasting")
plt.legend()
plt.show()
