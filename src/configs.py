from typing import Optional


class KMeansConfig:
    k: int = 2
    maxIter: int = 20
    seed: Optional[int] = None


class DataConfig:
    data_path: str = "data/openfood.csv"
    feature_path: str = "configs/features.json"


class SparkConfig:
    app_name: str = "food_cluster"
    deploy_mode: str = "local"
    driver_memory: str = "4g"
    executor_memory: str = "16g"
    executor_cores: int = 1
    driver_cores: int = 1


class TrainConfig:
    kmeans: KMeansConfig = KMeansConfig()
    data: DataConfig = DataConfig()
    spark: SparkConfig = SparkConfig()
    save_to: str = "models/food_cluster"
