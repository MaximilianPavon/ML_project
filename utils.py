import csv
import time

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
        predicted_labels = predicted_labels.astype(int)
    
    if not file_label == '':
        file_label += '-'
    
    with open('submissions/' + file_label + time.strftime("%Y-%m-%d-%H-%M-%S") + '.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Sample_id', 'Sample_label'])
        writer.writerows(enumerate(predicted_labels, start=1))
