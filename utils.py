import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

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

def plot_confusion_matrix(cm, classes=False,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not classes:
        classes = range(1, cm.shape[0] + 1)
    
    plt.figure(figsize=(11,6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./report_files/' + title.replace(" ", "_") + '.png', dpi=300, bbox_inches='tight');
