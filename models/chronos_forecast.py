
import os
from statistics import quantiles

# Set environment variables early, before any transformers/huggingface import
os.environ["HF_HOME"] = "/data2/amir/HF_home"

# os.environ["CUDA_HOME"] = "/data2/InstallFolder"
# os.environ["PATH"] = f"/data2/InstallFolder/bin:{os.environ['PATH']}"
# os.environ["LD_LIBRARY_PATH"] = f"/data2/InstallFolder/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
# os.environ["C_INCLUDE_PATH"] = f"/data2/InstallFolder/include:{os.environ.get('C_INCLUDE_PATH', '')}"
# os.environ["CPLUS_INCLUDE_PATH"] = f"/data2/InstallFolder/include:{os.environ.get('CPLUS_INCLUDE_PATH', '')}"
# os.environ["TRANSFORMERS_CACHE"] = "/data2/amir/HF_home/transformers"

import pandas as pd  # requires: pip install pandas
import torch
from chronos import BaseChronosPipeline
import matplotlib.pyplot as plt  # requires: pip install matplotlib


pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
    device_map="cpu",  # use "cpu" for CPU inference
    torch_dtype=torch.bfloat16,
)


def get_predictions_chronos(input_tensor: torch.Tensor, **kwargs):

    # context must be either a 1D tensor, a list of 1D tensors,
    # or a left-padded 2D tensor with batch as the first dimension
    # quantiles is an fp32 tensor with shape [batch_size, prediction_length, num_quantile_levels]
    # mean is an fp32 tensor with shape [batch_size, prediction_length]
    prediction_length = kwargs.get("prediction_length", 12)
    quantile_levels = kwargs.get("quantiles", [0.1, 0.5, 0.9])
    
    quantiles, mean = pipeline.predict_quantiles(
        context=input_tensor,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )

    return quantiles, mean


# forecast_index = range(len(df), len(df) + 12)
# low, median, high = quantiles[0, :, 0], quantiles[0, :, 1], quantiles[0, :, 2]

# plt.figure(figsize=(8, 4))
# plt.plot(df["#Passengers"], color="royalblue", label="historical data")
# plt.plot(forecast_index, median, color="tomato", label="median forecast")
# plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
# plt.legend()
# plt.grid()
# # plt.savefig("chronos_forecast.png")