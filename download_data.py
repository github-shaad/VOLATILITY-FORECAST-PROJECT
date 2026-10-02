import urllib.request
import os

os.makedirs("data", exist_ok=True)

DATA_URL = "https://drive.google.com/file/d/1pp90l8mAx8UzAhovuS_eG8-JlfpklfqT/view?usp=sharing"
DESTINATION = "data/nq_100ms_data.parquet"

print("Downloading dataset...")
urllib.request.urlretrieve(DATA_URL, DESTINATION)
print(f"Dataset successfully saved to {DESTINATION}")
