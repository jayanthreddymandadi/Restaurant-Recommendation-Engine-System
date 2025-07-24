import pandas as pd
from sklearn.cluster import KMeans

# STEP 1: Load data
test_locations = pd.read_csv('test_locations.csv')
vendors = pd.read_csv('vendors.csv')
orders = pd.read_csv('orders.csv', low_memory=False)  # suppress dtype warning

print("Test Locations Sample:")
print(test_locations.head())
print("Vendors Sample:")
print(vendors.head())

# STEP 2: Cluster test_locations (based on their own coordinates)
test_locations = test_locations.dropna(subset=['latitude', 'longitude'])
test_locations = test_locations[
    (test_locations['latitude'] > 0) & (test_locations['longitude'] > 0)
]

coords = test_locations[['latitude', 'longitude']]
kmeans = KMeans(n_clusters=50, random_state=42)
test_locations['cluster'] = kmeans.fit_predict(coords)

print("Test Locations with Cluster Labels:")
print(test_locations[['customer_id', 'location_number', 'cluster']].head())

# STEP 3: Rank top vendors per cluster based on past orders
# Merge using the correct location column from orders.csv: LOCATION_NUMBER
orders_with_cluster = orders.merge(
    test_locations[['customer_id', 'location_number', 'cluster']],
    left_on=['customer_id', 'LOCATION_NUMBER'],
    right_on=['customer_id', 'location_number'],
    how='inner'
)

# Count vendor orders per cluster
vendor_cluster_counts = (
    orders_with_cluster
    .groupby(['cluster', 'vendor_id'])
    .size()
    .reset_index(name='order_count')
)

# Rank vendors within each cluster
vendor_cluster_counts['rank'] = vendor_cluster_counts \
    .groupby('cluster')['order_count'] \
    .rank(ascending=False, method='first')

# Keep top N vendors per cluster
top_n = 5
top_vendors_per_cluster = vendor_cluster_counts[vendor_cluster_counts['rank'] <= top_n]

print("Top Vendors Per Cluster:")
print(top_vendors_per_cluster.head())
# STEP 4: Recommend top vendors to each test customer-location

# List to hold final results
recommendations = []

# Iterate over each test customer-location
for _, row in test_locations.iterrows():
    customer_id = row['customer_id']
    location_number = row['location_number']
    cluster = row['cluster']
    
    # Get top vendors for this cluster
    top_vendors = top_vendors_per_cluster[top_vendors_per_cluster['cluster'] == cluster] \
                    .sort_values('rank')['vendor_id'].tolist()
    
    # Create (customer_id, location_number, vendor_id) rows
    for vendor_id in top_vendors:
        recommendations.append([customer_id, location_number, vendor_id])

# Convert to DataFrame
submission_df = pd.DataFrame(recommendations, columns=['customer_id', 'location_number', 'vendor_id'])

# Preview
print(submission_df.head())

# Save to CSV
submission_df.to_csv('cluster_based_recommendations.csv', index=False)
print("✅ Recommendations saved to cluster_based_recommendations.csv")
# STEP 5: Format to match SampleSubmission style

# Create the combined key
submission_df['CID X LOC_NUM X VENDOR'] = (
    submission_df['customer_id'].astype(str) + ' X ' +
    submission_df['location_number'].astype(str) + ' X ' +
    submission_df['vendor_id'].astype(str)
)

# Keep only the required column
final_submission = submission_df[['CID X LOC_NUM X VENDOR']]

# Save to CSV
final_submission.to_csv('final_submission.csv', index=False)

print("✅ Final submission saved as final_submission.csv")
print(final_submission.head())
