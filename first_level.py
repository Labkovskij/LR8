import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Parse the files
try:
    data_description = pd.read_excel('data_description.xlsx')  # Змінено на локальний шлях
    sample_data = pd.read_excel('sample_data.xlsx')  # Змінено на локальний шлях
except Exception as e:
    print(f"Error reading files: {e}")
    raise

# Перевірка назв колонок у sample_data
print("Назви колонок у sample_data:")
print(sample_data.columns)

# Step 2: Select 17 indicators for the scoring table
desired_columns = [
    'loan_amount',
    'loan_days',
    'gender_id',
    'Marital status',
    'children_count_id',
    'education_id',
    'has_immovables',
    'has_movables',
    'employment_type_id',
    'position_id',
    'monthly_income',
    'monthly_expenses',
    'other_loans_active',
    'loan_closed',
    'loan_overdue',
    'product_id',
    'current_rest_amount'
]

# Перевірка наявності колонок
missing_columns = [col for col in desired_columns if col not in sample_data.columns]
if missing_columns:
    print(f"Відсутні колонки: {missing_columns}")
    raise KeyError(f"None of {missing_columns} are in the columns")

selected_data = sample_data[desired_columns]

# Step 3: Clean the data
cleaned_data = selected_data.fillna(selected_data.mean())

# Step 4: Calculate the integrated score (Scor)
weights = np.random.uniform(0.1, 1.0, len(desired_columns))
cleaned_data['Scor'] = cleaned_data.dot(weights)

# Step 5: Cluster borrowers based on binary decision
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cleaned_data)

kmeans = KMeans(n_clusters=2, random_state=42)
cleaned_data['Cluster'] = kmeans.fit_predict(scaled_data)

# Step 6: Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(cleaned_data['Scor'], cleaned_data['Cluster'], c=cleaned_data['Cluster'], cmap='viridis', alpha=0.7)
plt.title("Cluster Visualization Based on Scor")
plt.xlabel("Scor")
plt.ylabel("Cluster")
plt.colorbar(label="Cluster")
plt.grid(True)

# Збереження графіку в локальній директорії
plt.savefig('cluster_visualization.png')
plt.show()

# Save the processed data to a new Excel file
output_file = 'processed_data.xlsx'  # Збереження в локальній директорії
cleaned_data.to_excel(output_file, index=False)

print(f"Data processing and visualization completed. Processed data saved to {output_file}")