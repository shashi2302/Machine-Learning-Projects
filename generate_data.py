import numpy as np
import pandas as pd

def generate_synthetic_data(num_users=5, num_samples=100):
    for user in range(num_users):
        heart_rate = np.random.normal(70, 10, num_samples)  
        sleep_duration = np.random.uniform(4, 10, num_samples)  
        activity_level = np.random.randint(0, 10000, num_samples)  
        labels = np.random.randint(0, 2, num_samples) 
        df = pd.DataFrame({
            "heart_rate": heart_rate,
            "sleep_duration": sleep_duration,
            "activity_level": activity_level,
            "label": labels
        })
        df.to_csv(f"data/user_{user}_data.csv", index=False)

if __name__ == "__main__":
    generate_synthetic_data()
