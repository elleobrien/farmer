import pandas as pd

df = pd.read_csv("data_raw.csv")

all_features = df.columns

# Let's drop some features
names = [feat for feat in all_features if "net_name" in feat] # excluded for privacy reasons
useless = ["info_gew","info_resul","interviewtime","id","date"] # features that we expect are uninformative
drop_list = names + useless 

# Remove the questionnaire about agricultural practices until I can better understand it
practice_list = ["legum","conc","add","lact","breed","covman","comp","drag","cov","plow","solar","biog","ecodr"]
for feat in all_features:
    if any(x in feat for x in practice_list):
        drop_list.append(feat)


df = df.drop(columns=drop_list)

# Convert non-numeric features to numeric
non_numeric = list(df.select_dtypes(include=['O']).columns)
for col in non_numeric:
    codes,uniques=pd.factorize(df[col])
    df[col] = codes

df.to_csv("data_processed.csv")

