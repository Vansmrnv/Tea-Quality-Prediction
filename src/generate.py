import numpy as np 
import pandas as pd

# Number of attempts and features
num_attempts = 100
water_temperature = np.random.randint(60, 100, num_attempts)  # Random integers between 60 and 100
steeping_time = np.random.randint(1, 5, num_attempts)  # Random integers between 1 and 4

# Determine if the tea is made correctly based on the rules
tea_made_correctly = ((water_temperature >= 85) & (water_temperature <= 95) & 
                      (steeping_time >= 3) & (steeping_time <= 5)).astype(int)
# Create a DataFrame
data = pd.DataFrame({
    'water_temperature': water_temperature,
    'steeping_time': steeping_time,
    'tea_made_correctly': tea_made_correctly
})

# Save to CSV
data.to_csv('data/tea_data.csv', index=False)

# Print summary
print(data.describe())
