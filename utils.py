import time
import os
import pandas as pd
import numpy as np

def make_submission(predicted_labels, file_label='', log=False):
    """
    Parameters
    ----------
    predicted_labels: Array
        NumPy array with the predicted labels.
    file_label: str
        Optional string to label the resulting filename, in addition to the timestamp.
    """
    
    if log:
        file_label += '_log'
    
    if file_label:
        file_label += '-'
    
    if log:
        header = ['Sample_id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9','Class_10']
    else:
        header = ['Sample_id', 'Sample_label']
        assert all(predicted_labels % 1 == 0), 'Hey, you gave me a .something value! (integer or .0 assumed)'
        predicted_labels = predicted_labels.astype(int)
    
    subm_dir = 'submissions'
    if not os.path.exists(subm_dir):
        os.makedirs(subm_dir)
    
    filename = subm_dir + '/' + file_label + time.strftime("%Y-%m-%d-%H-%M-%S") + '.csv'
    df = pd.DataFrame(predicted_labels)
    df.index += 1
    df.to_csv(filename, index_label=header[0], header=header[1:])
