from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('customers').getOrCreate()

dataset = spark.read.csv("data/df_CatVal_cleanedV2.csv", header=True, inferSchema=True)
dataset.head(1)
# dataset.describe().show(1)
dataset.printSchema()

columns_to_drop = ['Address']
dataset = dataset.drop(*columns_to_drop)
dataset.printSchema()


from pyspark.ml.feature import VectorAssembler

dataset = dataset.withColumn("Defaulted", dataset["Defaulted"])

dataset.printSchema()

feat_cols = [
    'Age',
    'Edu', 'Years Employed', 'Income', 'Card Debt', 'Other Debt', 'DebtIncomeRatio']
vec_assembler = VectorAssembler(inputCols=feat_cols, outputCol='features')
final_data = vec_assembler.transform(dataset)
from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

scalerModel = scaler.fit(final_data)

cluster_final_data = scalerModel.transform(final_data)

kmeans3 = KMeans(featuresCol='scaledFeatures', k=3)
kmeans2 = KMeans(featuresCol='scaledFeatures', k=2)
kmeans1 = KMeans(featuresCol='scaledFeatures', k=1)

model3 = kmeans3.fit(cluster_final_data)
model2 = kmeans2.fit(cluster_final_data)
model = kmeans1.fit(cluster_final_data)

# Make predictions
predictions3 = model3.transform(cluster_final_data)
predictions2 = model2.transform(cluster_final_data)
# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions3)
print("With k=3 Silhouette with squared euclidean distance = " + str(silhouette))
silhouette = evaluator.evaluate(predictions2)
print("With k=2 Silhouette with squared euclidean distance = " + str(silhouette))

# Show the results
centers = model.clusterCenters()
print("Cluster Centers:")
for center in centers:
    print(center)

# Evaluate clustering by computing Within Set Sum of Squared Errors.
for k in range(2, 9):
    kmeans = KMeans(featuresCol='scaledFeatures', k=k)
    model = kmeans.fit(cluster_final_data)
    predictions = model.transform(cluster_final_data)
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    print("With K={}".format(k))
    print("Silhouette with squared euclidean distance = " + str(silhouette))
    print('--' * 30)

model3.transform(cluster_final_data).groupBy('prediction').count().show()

model2.transform(cluster_final_data).groupBy('prediction').count().show()
