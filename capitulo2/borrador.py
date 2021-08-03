from sklearn.svm import SVR
import os
import tarfile
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    #descarga el csv
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)# te lo carga a a disco, si quisieramos cargarlo directamente a ram usamos urllib.request.urlopen
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    #pasa el csv a pandas
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

fetch_housing_data()
rootHousingData = load_housing_data()
train_set, test_set = train_test_split(rootHousingData, test_size=0.2, random_state=42)
# print(len(test_set)/len(rootHousingData))


svr = SVR()
#svr.fit(train_set)
import matplotlib.pyplot as plt
import math
plt.hist(x=rootHousingData['median_house_value'], bins=9)
plt.savefig("mygraph.png")
categories =  pd.cut(x=rootHousingData['median_house_value'],bins=list(range(0,70000,math.floor(70000/9))),labels=list(range(0,9,1)))
print(categories)
# plt.hist(x=categories, bins=9)
# plt.savefig("cutgraph.png")

#plt.show()
