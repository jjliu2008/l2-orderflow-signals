import kagglehub

# Download latest version
path = kagglehub.dataset_download("wentinglu/highfrequency-futures-data-china")

print("Path to dataset files:", path)