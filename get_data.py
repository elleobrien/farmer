import os
import wget

# data from https://www.sciencedirect.com/science/article/pii/S2352340920303048
url ='https://shrutideveloper.s3.amazonaws.com/yshdbyj6zy-1.zip'
zip_name = "data.zip"

wget.download(url, zip_name)

#Now we will unzip the file
import zipfile
with zipfile.ZipFile(zip_name, 'r') as zip_ref:
    zip_ref.filelist[0].filename = 'data_raw.csv'
    zip_ref.extract(zip_ref.filelist[0])

os.remove(zip_name)