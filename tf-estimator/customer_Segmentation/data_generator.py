import numpy as np 
import pandas as pd 


#create dummy dataset
recency = np.random.randint(low=1, high=10, size=40000)
frequency = np.random.randint(low=1, high=10, size=40000)
monetary = np.random.randint(low=1, high=10, size=40000)

#create a dataframe
data = pd.DataFrame({"Recency":recency, "Frequency":frequency,
					"Monetary":monetary})

#convert into float
data["Recency"] = data["Recency"].astype(float)
data["Frequency"] = data["Frequency"].astype(float)
data["Monetary"] = data["Monetary"].astype(float)


#save the dataframe
data.to_csv("dummy_dataset_test.csv",sep=",")

