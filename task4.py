import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

np.random.seed(42)

data = {
    "Severity": np.random.choice([1,2,3,4], 500),
    "Weather": np.random.choice(["Clear","Rain","Fog","Snow"], 500),
    "Road_Condition": np.random.choice(["Dry","Wet","Icy"], 500),
    "Hour": np.random.randint(0,24,500),
    "Latitude": np.random.uniform(37.0, 38.0, 500),
    "Longitude": np.random.uniform(-122.0, -121.0, 500)
}

df = pd.DataFrame(data)

print("Dataset Shape:", df.shape)
print(df.head())


plt.figure(figsize=(8,5))
df["Hour"].value_counts().sort_index().plot(kind="line")
plt.title("Accidents by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Number of Accidents")
plt.show()


plt.figure(figsize=(6,4))
df["Weather"].value_counts().plot(kind="bar")
plt.title("Accidents by Weather Condition")
plt.show()


plt.figure(figsize=(6,4))
df["Road_Condition"].value_counts().plot(kind="bar")
plt.title("Accidents by Road Condition")
plt.show()


plt.figure(figsize=(6,4))
sns.countplot(x="Severity", data=df)
plt.title("Accident Severity Distribution")
plt.show()


plt.figure(figsize=(6,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
