import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# Load CSV
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)
df = df[df['gender'] != 'Other'].copy()
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')

# Spline Imputation
df.sort_values(by='age', inplace=True)
known = ~df['bmi'].isna()
spline = CubicSpline(df.loc[known, 'age'].unique(), df.groupby('age')['bmi'].mean(), extrapolate=True)
df.loc[~known, 'bmi'] = np.clip(spline(df.loc[~known, 'age']), 12, 65)

# Engineered Features
df['age_glucose'] = (df['age'] / 50.0) * (df['avg_glucose_level'] / 100.0)
df['vascular_score'] = df['hypertension'] * 1.5 + df['heart_disease'] * 2.0
df['age_bmi'] = (df['age'] / 50.0) * (df['bmi'] / 25.0)
df['high_glucose'] = (df['avg_glucose_level'] >= 140.0).astype(float)
df['elderly'] = (df['age'] >= 60.0).astype(float)

print("Engineered DataFrame Shape:", df.shape)
print("Features:", list(df.columns))
