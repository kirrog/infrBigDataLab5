from pyspark import SparkConf, SparkContext

file_path = "data/test_text_file.txt"
output_dir = "data/test_result"


def tokenize(line):
    return line.lower().split()


if __name__ == "__main__":
    conf = SparkConf().setAppName("WordCountApp").setMaster("local")
    sc = SparkContext(conf=conf)
    text_rdd = sc.textFile(file_path)
    words_rdd = text_rdd.flatMap(tokenize)
    word_pairs_rdd = words_rdd.map(lambda word: (word, 1))
    word_counts_rdd = word_pairs_rdd.reduceByKey(lambda a, b: a + b)
    # Sort by word count (descending)
    sorted_by_count_rdd = word_counts_rdd.sortBy(lambda x: x[1], ascending=False)
    # Sort alphabetically
    sorted_alphabetically_rdd = word_counts_rdd.sortBy(lambda x: x[0])
    sorted_by_count_rdd.saveAsTextFile(output_dir)
    # Stop the SparkContext
    sc.stop()
