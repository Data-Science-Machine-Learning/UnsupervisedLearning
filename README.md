# UnsupervisedLearning
This is unsupervised learning model to group or cluster given data in some class or group. Here, the kmean.csv file is the original data we have. It contains name, age, income fields. Assuming we don't have any target variable to predict and want to group them in some manner. For that we can use kmean model to make grouping of datasets and classify. Here in unsupervised_learning.py file, first we read csv with help of pandas library, then using MinMaxScaler to fit data appropriately in graph plot. Then we used kmean and feed age, income data to it to predict and classify them into some group (cluster). Once data classified we used matplot library to plot all separate clusters (df1, df2, df3) and also we plotted centroids (the point from where actual data points are closer to it). If you have noticed we have taken n_clusters = 3, which comes from elbow technique mentioned in the code in commented form at last. Elbow technique suggests us to choose this cluster value (K value), means in how many cluster or group our data can be classified for best result. Have uploaded output.png and elbow_selectKvalue.png files to show outputs in better way.
