import geopy.distance
import pandas as pd

df = pd.read_csv("dataset/JC-20162-citibike-tripdata.csv", sep=",")
print(df.shape)

def calc_distance(df):
	dist = []
	
	#calculate the Start Coordinates
	df['Start Coordinates'] = df['Start Station Latitude']/df['Start Station Longitude']
	
	#calculate the End coordinates
	df['End Coordinates'] = df['End Station Latitude']/df['End Station Longitude']
	
	for i in range(df.shape[0]):
		dist.append(geopy.distance.vincenty(
				df.iloc[i]['Start Coordinates'],
				df.iloc[i]['End Coordinates']
				).miles)
	return dist

distance = calc_distance(df)
df = pd.DataFrame({'distance':distance})

df.to_csv("distance.csv", sep=",")
