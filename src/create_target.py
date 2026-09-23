import pandas as pd

INPUT_FILE = "data/raw/olist_orders_dataset.csv"
OUTPUT_FILE = "data/processed/orders_with_target.csv"

# Load orders
df = pd.read_csv(INPUT_FILE)

# Keep delivered orders only
df = df[df["order_status"] == "delivered"].copy()

# Convert delivery dates
df["order_delivered_customer_date"] = pd.to_datetime(
    df["order_delivered_customer_date"]
)

df["order_estimated_delivery_date"] = pd.to_datetime(
    df["order_estimated_delivery_date"]
)

# Remove orders without actual delivery date
df = df.dropna(subset=["order_delivered_customer_date"])

# Create target
df["delivered_late"] = (
    df["order_delivered_customer_date"]
    > df["order_estimated_delivery_date"]
).astype(int)

# Save processed data
df.to_csv(OUTPUT_FILE, index=False)

print(f"Processed orders: {len(df)}")
print("\nDelivery target distribution:")
print(df["delivered_late"].value_counts())

print("\nDelivery target percentage:")
print(df["delivered_late"].value_counts(normalize=True) * 100)