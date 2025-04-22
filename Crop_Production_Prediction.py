from altair import Element
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Data Cleaning and Preprocessing

# Load the dataset
file_path = "FAOSTAT_data.csv"
df = pd.read_csv(file_path)

# Fill or drop missing values (customize as needed)
df.fillna(0, inplace=True)  

# drop duplicates
df.drop_duplicates(inplace=True)

# Save cleaned data to a new CSV file
df.to_csv("Cleaned_FAOSTAT_data.csv", index=False)

# ------------------------------------------------------------------------------------------------------

# Exploratory Data Analysis

# Crop Frequency Distribution

plt.figure(figsize=(12, 5))
sns.countplot(y=df["Item"], order=df["Item"].value_counts().index)
plt.title("Most Cultivated Crops")
plt.xlabel("Count")
plt.ylabel("Crop Type")
plt.show()

# Geographical Distribution

plt.figure(figsize=(12, 5))
sns.countplot(y=df["Area"], order=df["Area"].value_counts().head(10).index)
plt.title("Top 10 Agricultural Regions")
plt.xlabel("Count")
plt.ylabel("Region")
plt.show()

# Yearly Trends

plt.figure(figsize=(12, 5))
sns.lineplot(x="Year", y="Value", hue="Element", data=df)
plt.title("Yearly Trends in Area Harvested, Yield, and Production")
plt.xlabel("Year")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.legend(title="Element")
plt.show()

# Growth Analysis (Trends in Yield & Production)

# Filter for Yield and Production data
filtered_df = df[df['Element'].isin(['Yield', 'Production'])]

# Group by crop and year, then calculate average yield and production trends
growth_trends = filtered_df.groupby(['Item', 'Year', 'Element'])['Value'].mean().reset_index()

# Plot trends
plt.figure(figsize=(12, 6))
sns.lineplot(x="Year", y="Value", hue="Item", data=growth_trends[growth_trends['Element'] == 'Production'])
plt.title("Crop Production Trends Over Time")
plt.xlabel("Year")
plt.ylabel("Production (tons)")
plt.legend(title="Crop Type", bbox_to_anchor=(1, 1))
plt.xticks(rotation=45)
plt.show()

# Environmental Relationships (Inferring Productivity)

# Pivot data to analyze Area harvested vs Yield
df_pivot = df.pivot_table(index=['Area', 'Year'], columns='Element', values='Value').reset_index()

# Scatter plot to visualize the relationship
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df_pivot["Area harvested"], y=df_pivot["Yield"])
plt.title("Impact of Area Harvested on Yield")
plt.xlabel("Area Harvested (ha)")
plt.ylabel("Yield (kg/ha)")
plt.show()

#  Input-Output Relationships (Land Usage & Productivity)

# Correlation between Area Harvested, Yield, and Production
corr_matrix = df_pivot[['Area harvested', 'Yield', 'Production']].corr()

# Heatmap to show relationships
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Area, Yield, and Production")
plt.show()

#  Comparative Analysis (High-Yield vs Low-Yield Crops)

# Compute average yield per crop
crop_yield = df[df['Element'] == 'Yield'].groupby('Item')['Value'].mean().reset_index()

# Sort and plot
crop_yield = crop_yield.sort_values(by='Value', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(y="Item", x="Value", data=crop_yield.head(10), palette="viridis")
plt.title("Top 10 High-Yield Crops")
plt.xlabel("Average Yield (kg/ha)")
plt.ylabel("Crop")
plt.show()

# Comparative Analysis Across Regions

# Compute average production per region
region_production = df[df['Element'] == 'Production'].groupby('Area')['Value'].sum().reset_index()

# Sort and plot
region_production = region_production.sort_values(by='Value', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(y="Area", x="Value", data=region_production.head(10), palette="magma")
plt.title("Top 10 Regions by Crop Production")
plt.xlabel("Total Production (tons)")
plt.ylabel("Region")
plt.show()

# Productivity Analysis

# Calculate productivity: Production per Area harvested
df_pivot['Productivity'] = df_pivot['Production'] / df_pivot['Area harvested']

# Plot productivity across regions
plt.figure(figsize=(12, 6))
sns.boxplot(x="Productivity", y="Area", data=df_pivot)
plt.title("Productivity Comparison Across Regions")
plt.xlabel("Production per Hectare")
plt.ylabel("Region")
plt.show()

# Outlier Detection in Yield & Production

plt.figure(figsize=(12, 6))
sns.boxplot(x="Element", y="Value", data=df[df["Element"].isin(["Yield", "Production"])])
plt.title("Outliers in Yield and Production")
plt.xlabel("Metric")
plt.ylabel("Value")
plt.show()

# ------------------------------------------------------------------------------------------------------

# Keep relevant columns

df = df[['Area', 'Item', 'Year','Element','Value']]
df = df[df['Element'].isin(['Area harvested','Yield', 'Production'])]

# Pivot the dataframe to create separate columns for each Element

df_pivot = df.pivot_table(index=['Area', 'Item', 'Year'], columns='Element', values='Value', aggfunc='mean').reset_index()

df_pivot['Production'].fillna(df_pivot['Production'].median(), inplace=True)

# Rename columns for clarity
df_pivot = df_pivot.rename(columns={'Production': 'Production', 'Yield': 'Yield', 'Area harvested': 'Area_harvested'})

# Define features and target variable
X = df_pivot.drop(columns=['Production'])
X.dropna()
X.fillna(0, inplace=True)
# X.fillna(X.mean(), inplace=True)
y = df_pivot['Production']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Identify categorical and numerical columns
cat_features = ['Area', 'Item']
num_features = ['Area_harvested', 'Yield']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

# Fit on training data
pipeline.fit(X_train, y_train)

# Save pipeline
joblib.dump(pipeline, "pipeline_model.pkl")

# Save column names used for raw input (X_train.columns)
joblib.dump(X_train.columns.tolist(), "input_features.pkl")

# Fit and transform training data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Initialize and train model
model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

