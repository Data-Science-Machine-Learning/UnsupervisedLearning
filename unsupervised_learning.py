import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# kmean use
df = pd.read_csv('kmean.csv')
plt.style.use('fivethirtyeight')

scaler = MinMaxScaler()
scaler.fit(df[['age']])
df['age'] = scaler.transform(df[['age']])
scaler.fit(df[['income']])
df['income'] = scaler.transform(df[['income']])

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['age', 'income']])
df['cluster'] = y_predicted
print(df)
print('Cluster Centroids: '+str(km.cluster_centers_))

# separate data frames for each cluster
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

# plotting each data frame
plt.scatter(df1.age, df1.income, label='Income', color='red')
plt.scatter(df2.age, df2.income, label='Income', color='green')
plt.scatter(df3.age, df3.income, label='Income', color='blue')

# plotting centroids
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], label='Centroid', color='black', marker='x')

plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Cluster & Centroid')
plt.legend()
plt.show()

# selecting value of k with elbow technique
# k_range = range(1, 10)
# sse = []
# for k in k_range:
#     km = KMeans(n_clusters=k)
#     km.fit(df[['age', 'income']])
#     temp_sse = km.inertia_  # sum of squared errors
#     sse.append(temp_sse)
#
# plt.plot(k_range, sse)
# plt.xlabel('K')
# plt.ylabel('SSE')
# plt.title('Elbow Technique')
# plt.show()
