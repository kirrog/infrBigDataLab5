from pyspark import SparkContext
from pyspark.sql.functions import count, when, isnan

spark = SparkContext(__name__)

data = spark.read.csv("local", "data/en_openfoodfacts_org_products.csv", sep="\t", encoding="utf-8", header='true').limit(1000)

num_of_rows = data.count()
num_of_columns = len(data.columns)

print(f"Dataset consists of {num_of_rows} lines, {num_of_columns} variables")


def null_factor(df, tx_threshold=50):
    nulll_rate = data.select([count(when(isnan(c), c)).alias(c) / num_of_rows * 100 for c in data.columns])

    null_rate.columns = ['Variable', 'Num_of_null']
    high_null_rate = null_rate[null_rate.Num_of_null >= tx_threshold]
    return high_null_rate
