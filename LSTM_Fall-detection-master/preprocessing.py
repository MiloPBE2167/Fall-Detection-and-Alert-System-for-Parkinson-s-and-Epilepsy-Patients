import numpy as np
import pandas as pd
import os


number_video = 128
sequence_length = 25
Duongdanlink = 'txtpre-processing'

actions = ['falltxt','lyingtxt','sittxt','standtxt','walktxt']
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    number = 1
    for sequence in np.array(os.listdir(os.path.join(Duongdanlink, action))):
        if number <=128:
            window = []
            data = pd.read_csv(os.path.join(Duongdanlink, action, str(sequence)))
            dataset = data.iloc[:,1:].values
            window.append(dataset)
            sequences.append(window)
            number +=1 
            labels.append(label_map[action])
print(np.array(sequences).shape)
