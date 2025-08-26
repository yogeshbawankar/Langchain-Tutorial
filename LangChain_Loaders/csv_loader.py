from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path = 'Social_Network_Ads.csv')

data = loader.load()

# It retrieve the data from CSV file row wise data[0] = row 1 data
print(data[0])