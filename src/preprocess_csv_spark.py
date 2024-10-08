import json

import pyspark.sql
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import functions

FEATURES_COLUMN = "scaled_feature"


class Preprocessor:
    def __init__(self, spark_cs: pyspark.sql.SparkSession, features_path: str):
        with open(features_path, "r") as features_file:
            self.features = json.load(features_file)
        self.spark = spark_cs

    def load_data(self, data_path: str) -> pyspark.sql.DataFrame:
        df = self.spark.read.csv(data_path, header=True, inferSchema=True, encoding="utf-8")
        return df

    def preprocess(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        id_columns = self.features["id"]
        feature_numeric = self.features["numeric"]
        numeric_columns = [
            functions.col(c).cast("float").alias(c) for c in feature_numeric
        ]
        cat_columns = self.features["categorical"]

        all_columns = id_columns + numeric_columns + cat_columns
        df_with_selected_columns = df.select(*all_columns)

        vec_assembler = VectorAssembler(
            inputCols=feature_numeric, outputCol="features"
        )
        df_with_features = vec_assembler.transform(df_with_selected_columns)

        scaler = StandardScaler(inputCol="features", outputCol=FEATURES_COLUMN)
        scaler_model = scaler.fit(df_with_features)
        df_scaled_features = scaler_model.transform(df_with_features)
        return df_scaled_features
