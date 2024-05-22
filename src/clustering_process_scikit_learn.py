import joblib
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

nltk.download('stopwords')
nltk.download('punkt')

df = pd.read_csv("../data/df_CatVal_cleanedV2.csv",
                 encoding="utf-8", low_memory=False, index_col=0)



df_init = df

for i in range(len(df.columns)):
    print(df.dtypes.index[i], ' : ', df.dtypes[i])

col_names = []
for x in df.columns:
    col_names.append(x)

index = 0
for i in (df.loc[6]):
    print(str(col_names[index]) + " : " + str(i))
    index += 1

# df['pnns1'] = pd.factorize(df['pnns1'])[0]

numeric_cols = list(df.select_dtypes(include=["float64", "int64"]).columns)

scaler = StandardScaler()

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

cols_to_convert_int = list(df.select_dtypes(include=["int64"]).columns)
cols_to_convert_float = list(df.select_dtypes(include=["float64"]).columns)

df[cols_to_convert_int] = df[cols_to_convert_int].astype('int32')
df[cols_to_convert_float] = df[cols_to_convert_float].astype('float32')

# numeric_cols.remove('pnns1')
numeric_cols.remove('nutriscore')
numeric_cols.remove('unique_scans_n')
numeric_cols.remove('completeness')

X = df[numeric_cols]

y = df['product_name']

pca = PCA(n_components=20)
pca_df = pd.DataFrame(pca.fit_transform(df[numeric_cols]))

from sklearn.cluster import KMeans

k = 10
k_values = range(1, k + 1)
inertias = []
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pca_df)
    inertias.append(kmeans.inertia_)

plt.plot(k_values, inertias, 'bx-')
plt.xlabel('Num of clusters')
plt.ylabel('Inertion')
plt.title("Search")
plt.show()

kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pca_df)

df['cluster_K_pca'] = kmeans.labels_

print(kmeans.inertia_)

joblib.dump(kmeans, '../model/kmeans_pca.pkl')

# kmeans = joblib.load('kmeans_pca.pkl')
