import os
from itertools import combinations
import pandas as pd
import multiprocessing
import time
from tqdm import tqdm
import argparse
from pyspark.sql import SparkSession

from pyspark.sql.functions import udf
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import PandasUDFType

from rapidfuzz.distance import Levenshtein


from pyspark.sql.types import IntegerType

def edit_distance(pair):
    str1, str2 = pair
    m, n = len(str1), len(str2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j 
            elif j == 0:
                dp[i][j] = i  
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],    
                                   dp[i - 1][j],    
                                   dp[i - 1][j - 1])  

    return dp[m][n]


def compute_edit_distance_multiprocess(pair, num_workers):

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(edit_distance, pair), total=len(pair), ncols= 100 ,desc="Multiprocessing"))
    return results





def compute_edit_distance_spark(pairs):


    spark = SparkSession.builder \
        .appName("BirdFlockSimulation") \
        .master("local[*]") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    df = spark.createDataFrame(pairs, ["str1", "str2"])



    start_time = time.time()


    @pandas_udf(IntegerType(), PandasUDFType.SCALAR)
    def edit_distance_udf(s1: pd.Series, s2: pd.Series) -> pd.Series:
        distances = [Levenshtein.distance(str1, str2) for str1, str2 in zip(s1, s2)]
        return pd.Series(distances)

    distances = df.withColumn("edit_distance", edit_distance_udf(df.str1, df.str2)).select("edit_distance").collect()
    end_time = time.time()

    edit_distances = [row['edit_distance'] for row in distances]

    spark.stop()

    return edit_distances, end_time - start_time





if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Edit Distance with PySpark")
    parser.add_argument('--csv_dir', type=str, default='simple-wiki-unique-has-end-punct-sentences.csv', help="Directory of csv file")
    parser.add_argument('--num_sentences', type=int, default=300, help="Number of sentences")
    args = parser.parse_args()

    numWorkers = multiprocessing.cpu_count()
    print(f'number of available cpu cores: {numWorkers}')

    text_data = pd.read_csv(args.csv_dir)['sentence']
    text_data = text_data[:args.num_sentences]
    pair_data = list(combinations(text_data, 2))

    print("Initialized Spark Session")
    sparkDistances, sparkTime = compute_edit_distance_spark(pair_data)
    print(f"Time taken: {sparkTime:.3f} seconds")

    startTime = time.time()
    print("Multi-Process Computation")
    editDistances = compute_edit_distance_multiprocess(pair_data, numWorkers)

    multiprocessTime = time.time() - startTime

    print(f"Time taken (multi-process): {multiprocessTime} seconds")

    startTime = time.time()
    distances = []

    for pair in tqdm(pair_data, ncols=100):
        distances.append(edit_distance(pair))

    forLoopTime = time.time() - startTime

    print(f"Time taken (for-loop): {forLoopTime} seconds")
    print(f"Time cost (Spark, multi-process, for-loop): [{sparkTime:.3f}, {multiprocessTime:.3f}, {forLoopTime:.3f}]")