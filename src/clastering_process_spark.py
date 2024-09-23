import logging

from pyspark.ml import clustering, evaluation
from pyspark.sql import SparkSession

from src import configs, preprocess_csv_spark

logger = logging.Logger("clustering")

def run(config: configs.TrainConfig):
    spark_config = config.spark
    spark = (
        SparkSession.builder.appName(spark_config.app_name)
        .master(spark_config.deploy_mode)
        .config("spark.driver.cores", spark_config.driver_cores)
        .config("spark.executor.cores", spark_config.executor_cores)
        .config("spark.driver.memory", spark_config.driver_memory)
        .config("spark.executor.memory", spark_config.executor_memory)
        .getOrCreate()
    )

    preprocessor = preprocess_csv_spark.Preprocessor(spark, config.data.feature_path)
    data = preprocessor.load_data(config.data.data_path)
    df = preprocessor.preprocess(data)

    kmeans_kwargs = config.kmeans.__dict__
    logger.info("Using kmeans model with parameters: {}", kmeans_kwargs)
    logger.info("Training")
    model = clustering.KMeans(featuresCol=preprocess_csv_spark.FEATURES_COLUMN, **kmeans_kwargs)
    model_fit = model.fit(df)

    logger.info("Evaluation")
    evaluator = evaluation.ClusteringEvaluator(
        predictionCol="prediction",
        featuresCol=preprocess_csv_spark.FEATURES_COLUMN,
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )
    output = model_fit.transform(df)
    output.show()

    score = evaluator.evaluate(output)
    logger.info("Silhouette Score: {}", score)

    logger.info("Saving to {}", config.save_to)
    model_fit.write().overwrite().save(config.save_to)
