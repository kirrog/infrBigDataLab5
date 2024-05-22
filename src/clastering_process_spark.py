import json

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('customers').getOrCreate()

dataset = spark.read.csv("data/df_CatVal_cleanedV2.csv", header=True, inferSchema=True)
print(dataset.head(1))
# dataset.describe().show(1)
dataset.printSchema()

vec_assembler = VectorAssembler(inputCols=dataset.columns, outputCol='features')
final_data = vec_assembler.transform(dataset)

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

scalerModel = scaler.fit(final_data)

cluster_final_data = scalerModel.transform(final_data)

results = dict()
# Evaluate clustering by computing Within Set Sum of Squared Errors.
for k in range(2, 30):
    kmeans = KMeans(featuresCol='scaledFeatures', k=k)
    model = kmeans.fit(cluster_final_data)
    predictions = model.transform(cluster_final_data)
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    results[k] = silhouette
    print("With K={}".format(k))
    print("Silhouette with squared euclidean distance = " + str(silhouette))
    print('--' * 30)

with open("metrics/kmeans_metrics.json", "w", encoding="UTF-8") as f:
    json.dump(results, f, ensure_ascii=False)
