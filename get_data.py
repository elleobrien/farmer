import os
import wget

# data from https://www.sciencedirect.com/science/article/pii/S2352340920303048

# Download the zipped dataset
url = 'https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/383116/rawdata_new.csv?sequence=1&isAllowed=y'
print(url)
zip_name = "data_raw.csv"
wget.download(url, zip_name)

