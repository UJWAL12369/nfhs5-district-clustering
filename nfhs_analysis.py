import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv('NFHS_5.csv')
print(df)

print(df.shape)

print(df.columns)
print(df.info())
print(df.describe())

print(df.isnull().sum())

df = df.drop_duplicates()


df=df.dropna()

print(df.shape)

# strrip spaces from column names
df.columns = df.columns.str.strip()
print(df.columns)


print(df.dtypes)



# Count how many cells contain *
star_count = (df.astype(str).apply(lambda col: col.str.contains(r"\*", regex=True))).sum()

print(star_count[star_count > 0])


import numpy as np

df = df.replace(r'^\*+$', np.nan, regex=True)


for col in df.columns[2:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')


print(df.isnull().sum().sort_values(ascending=False))


# Drop any column with more than 30% missing values.
threshold = len(df) * 0.30

df = df.dropna(axis=1, thresh=len(df)-threshold)

print(df.shape)

print(df.isnull().sum().sort_values(ascending=False))


numeric_cols = df.select_dtypes(include='number').columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

print(df.isnull().sum().sort_values(ascending=True).head())
print(df.shape)



numeric_cols = df.select_dtypes(include='number').columns

print("Number of numeric columns:", len(numeric_cols))

categorical_cols = df.select_dtypes(exclude='number').columns

print(categorical_cols)



# Separate Identifier Columns

id_cols = ['State/UT', 'District Names']   # Change names if your dataset differs

id_data = df[id_cols]


# Select only numerical columns
X = df.select_dtypes(include=['number']).copy()

print("Number of features:", X.shape[1])
print(X.head())

# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.figure(figsize=(18,15))

# corr = X.corr()

# sns.heatmap(corr,
#             cmap='coolwarm',
#             center=0)

# plt.title("Correlation Heatmap")
# plt.show()


# Remove Highly Correlated Variables
corr_matrix = X.corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

print("Columns to drop:")
print(to_drop)

X = X.drop(columns=to_drop)

print("Remaining Features:", X.shape)


# Standardize Features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



# elbow method

from sklearn.cluster import KMeans

# wcss = []

# for k in range(2,11):

#     km = KMeans(
#         n_clusters=k,
#         random_state=42,
#         n_init=10
#     )

#     km.fit(X_scaled)

#     wcss.append(km.inertia_)

# plt.figure(figsize=(8,5))

# plt.plot(range(2,11),
#          wcss,
#          marker='o')

# plt.xlabel("Number of Clusters")

# plt.ylabel("WCSS")

# plt.title("Elbow Method")

# plt.grid()

# plt.show()



kmeans = KMeans(
    n_clusters=6,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)



print(df["Cluster"].value_counts())


# plt.figure(figsize=(6,4))

# sns.countplot(
#     x="Cluster",
#     data=df,
#     palette="Set2"
# )

# plt.title("Number of Districts in Each Cluster")

# plt.show()


# cluster summary

cluster_summary = df.groupby("Cluster")[X.columns].mean()

print(cluster_summary)

# cluster_summary = df.groupby("Cluster")[X.columns].mean().round(2)

# cluster_summary.to_csv("Cluster_Summary.csv")

# print("Cluster summary saved successfully.")



# pca

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:,0]
df["PCA2"] = X_pca[:,1]





plt.figure(figsize=(10,7))

sns.scatterplot(
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    data=df,
    palette="Set1",
    s=80
)

plt.title("District Clusters using PCA")

plt.show()


cluster_names = {
    0: "Moderate Health Districts",
    1: "Nutritionally Vulnerable Districts",
    2: "Lifestyle & Health Risk Districts",
    3: "Urban Transition Districts",
    4: "High Child Health Risk Districts",
    5: "High Performing Health Districts"
}

df["Cluster_Name"] = df["Cluster"].map(cluster_names)


plt.figure(figsize=(10,5))

sns.countplot(
    x="Cluster_Name",
    data=df,
    order=df["Cluster_Name"].value_counts().index
)

plt.xticks(rotation=20)
plt.title("Distribution of Districts Across Health Clusters")
plt.xlabel("Health Cluster")
plt.ylabel("Number of Districts")

plt.show()


# heatmap
# plt.figure(figsize=(22,18))

# sns.heatmap(
#     cluster_summary.T,
#     cmap="RdYlGn",
#     center=cluster_summary.values.mean()
# )

# plt.title("Cluster-wise Average of NFHS Indicators")

# plt.xlabel("Cluster")
# plt.ylabel("Indicators")

# plt.show()


df.to_csv("NFHS_District_Clusters.csv", index=False)