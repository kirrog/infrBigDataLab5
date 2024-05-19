from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName("WordCountApp").setMaster("local")
sc = SparkContext(conf=conf)

file_path = "data/test_text_file.txt"
text_rdd = sc.textFile(file_path)


def tokenize(line):
    return line.lower().split()


words_rdd = text_rdd.flatMap(tokenize)

word_pairs_rdd = words_rdd.map(lambda word: (word, 1))
word_counts_rdd = word_pairs_rdd.reduceByKey(lambda a, b: a + b)

# Sort by word count (descending)
sorted_by_count_rdd = word_counts_rdd.sortBy(lambda x: x[1], ascending=False)

# Sort alphabetically
sorted_alphabetically_rdd = word_counts_rdd.sortBy(lambda x: x[0])

output_dir = "data/test_result"
sorted_by_count_rdd.saveAsTextFile(output_dir)

# Stop the SparkContext
sc.stop()