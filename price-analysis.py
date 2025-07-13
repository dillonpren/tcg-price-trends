import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# import matplotlib.pyplot as plt

from card_stats import card_stats as cs
from events import events as events
from sets import sets as s1
from sales import sales as s2


def get_event_impact(date):
    window = event_influences[(event_influences['Date'] >= date - pd.Timedelta(days=30)) &
                               (event_influences['Date'] <= date + pd.Timedelta(days=30))]
    if window.empty:
        return 'None'
    # Take most impactful if multiple
    impact_order = ['Positive', 'Neutral', 'Negative']
    for impact in impact_order:
        if impact in window['Likely_Impact'].values:
            return impact
    return 'None'

# Load data
card_stats = pd.read_csv("card_stats.csv")
event_influences =  pd.read_csv("/event_influences.csv")
set_metadata = pd.read_csv("set_metadata.csv")
set_sales =  pd.read_csv("set_sales.csv")

# Encode Meta_Relevance ordinally
meta_map = {'Low': 0, 'Medium': 1, 'High': 2}
card_stats['Meta_Score'] = card_stats['Meta_Relevance'].map(meta_map)

# Encode Is_Reprint
card_stats['Is_Reprint_Binary'] = card_stats['Is_Reprint'].map({'Yes': 1, 'No': 0})

# Rarity mapping (simplified scale)
rarity_map = {
    'Common': 1, 'Rare': 2, 'Super Rare': 3, 'Ultra Rare': 4, 'Secret Rare': 5
}
card_stats['Rarity_Score'] = card_stats['Rarity'].map(rarity_map)

# Aggregate card stats per Set_Code
card_agg = card_stats.groupby('Set_Code').agg({
    'Meta_Score': 'mean',
    'Is_Reprint_Binary': 'mean',
    'Rarity_Score': 'mean',
    'Card_Name': 'count'  # proxy for number of cards
}).rename(columns={'Card_Name': 'Num_Cards'}).reset_index()

# Merge all data
merged = set_sales.merge(set_metadata, left_on='Code', right_on='Code')
merged = merged.merge(card_agg, left_on='Code', right_on='Set_Code', how='left')

# Convert Release_Date to datetime
merged['Release_Date'] = pd.to_datetime(merged['Release_Date'])

# Event influence flags (simplified approach: check if an event occurred within +/-30 days of release)
event_influences['Date'] = pd.to_datetime(event_influences['Date'])

merged['Event_Impact'] = merged['Release_Date'].apply(get_event_impact)

# Encode categorical features
le = LabelEncoder()
merged['Event_Impact_Encoded'] = le.fit_transform(merged['Event_Impact'])
merged['Product_Category_Encoded'] = le.fit_transform(merged['Product_Category'])

# Drop missing values
merged = merged.dropna(subset=['Meta_Score', 'Rarity_Score'])

# -----------------------
# Modeling
# -----------------------

features = ['Meta_Score', 'Is_Reprint_Binary', 'Rarity_Score', 'Num_Cards',
            'Event_Impact_Encoded', 'Product_Category_Encoded']
X = merged[features]
y = merged['Revenue_USD']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Random forest regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# R² score
r2 = r2_score(y_test, y_pred)

# Feature importances
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

# Print R² score and top features
print(f"Model R² Score: {r2:.2f}")
print("\nFeature Importance (descending):")
for feature, importance in importances.items():
    print(f"{feature}: {importance:.4f}")