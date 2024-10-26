import pandas as pd
import os


directory = 'txtpre-processing/walktxt'
number = 1

dfs = []


for filename in os.listdir(directory):
    while number <=128:
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
        
            dfs.append(pd.read_csv(filepath))
            number +=1

# Concatenate all the dataframes in the list
combined_df = pd.concat(dfs, ignore_index=True)

# Save the concatenated dataframe to a new CSV file
combined_df.to_csv('WALK.txt', index=False)