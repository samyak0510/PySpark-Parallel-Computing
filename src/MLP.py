import os
os.system('clear')
import argparse
import torch
import torch.nn as nn
import time
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql import SparkSession
from pyspark.broadcast import Broadcast
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F



class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims):
        super(MLPClassifier, self).__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, num_classes))  
        self.model = nn.Sequential(*layers)

    def forward(self, x):

        logits = self.model(x)

        predicted_classes = torch.argmax(logits, dim=1)
        return predicted_classes


def get_model(input_dim, num_classes, hidden_dims):

    mlp_model = MLPClassifier(input_dim, num_classes, hidden_dims)
    mlp_model.eval()

    return mlp_model




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Edit Distance with PySpark")
    parser.add_argument('--n_input', type=int, default=10000, help="Number of sentences")
    parser.add_argument('--hidden_dim', type=int, default=1024, help="hidden_dim")
    parser.add_argument('--hidden_layer', type=int, default=50, help="hidden_layer")
    args = parser.parse_args()

    inputDim = 128  
    numClasses = 10  

    hiddenDims = [args.hidden_dim for _ in range(args.hidden_layer)]  


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"UsingDevice: {device}")


    x = torch.randn(args.n_input, inputDim).to(device)

    mlpModel = get_model(inputDim, numClasses, hiddenDims).to(device)
    mlpModel.eval()

    startTime = time.time()
    with torch.no_grad():
        output = mlpModel(x)
    endTime = time.time()

    nonSparkTime = endTime - startTime

    print(f"Output: {output.shape}")

    print(f"NonSparkInferenceTime: {nonSparkTime:.6f} seconds")

    xCpu = x.cpu().numpy()

    inputDf = pd.DataFrame(xCpu.tolist())

    # spark = SparkSession.builder \
    #     .appName("MLPInferenceSpark") \
    #     .config("spark.driver.memory", "4g") \
    #     .getOrCreate()
    # sc = spark.sparkContext

    
    spark = SparkSession.builder \
        .appName("BirdFlockSimulation") \
        .master("local[*]") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    sc = spark.sparkContext

    sparkDf = spark.createDataFrame(inputDf)

    modelState = mlpModel.to('cpu').state_dict()
    modelBc = sc.broadcast(modelState)

    @pandas_udf(IntegerType())
    def predictUdf(*cols):
          import torch
          from pandas import DataFrame

          inputs = torch.tensor(DataFrame(cols).T.values, dtype=torch.float32)
          model = MLPClassifier(inputDim, numClasses, hiddenDims)
          model.load_state_dict(modelBc.value)

          model.eval()
          with torch.no_grad():
              outputs = model(inputs)
          return pd.Series(outputs.numpy())

    startTime = time.time()
    cols = sparkDf.columns
    predictions = sparkDf.select(predictUdf(*[F.col(c) for c in cols]).alias("prediction"))

    predictions.collect()  
    endTime = time.time()
    sparkTime = endTime - startTime

    print(f"Spark Inference Time: {sparkTime:.6f} seconds")
    spark.stop()

    print(f"Output: {output.shape}")
    print(f"Time taken for Forward Pass: {nonSparkTime:.6f} seconds")
    print(f"Time cost for spark and non spark version: [{sparkTime:.6f},  {nonSparkTime:.6f}] seconds")
