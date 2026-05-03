import urllib.request
import zipfile
import os

url = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
file_name = "ml-20m.zip"

print("Downloading dataset...")
urllib.request.urlretrieve(url, file_name)

print("Extracting...")
with zipfile.ZipFile(file_name, 'r') as zip_ref:
    zip_ref.extractall("data")

os.remove(file_name)
print("Done.")
