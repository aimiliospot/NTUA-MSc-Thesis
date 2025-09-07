import os
import shutil
import random
from PIL import Image
import pandas as pd

def trainingImagesSelection(train_df) :
    try :
        shutil.rmtree('trainDataset')
    except :
        print('No such directory')
    number_of_classes = len(os.listdir('CovertedDicomToPng/'))
    for i in range(1,number_of_classes + 1):
        os.makedirs('trainDataset/class_'+str(i))
    min = 9e6
    class_min = 0
    for i in range(1,number_of_classes + 1):
        path = 'CovertedDicomToPng/class_' + str(i)
        if len(os.listdir(path)) < min :
            min = len(os.listdir(path))
            class_min = i
    for i in range(1,number_of_classes + 1):
        path = 'CovertedDicomToPng/class_' + str(i)
        file_list = os.listdir(path)
        selected_files = random.sample(file_list, min)
        for file in selected_files:
            source_path = os.path.join(path, file)
            destination_path = os.path.join('trainDataset/class_'+str(i), file)
            shutil.copy(source_path, destination_path)
            im = Image.open(destination_path)
            im = im.rotate(180)
            destination_path = os.path.join('trainDataset/class_'+str(i), 'rotated-'+file)
            im.save(destination_path)
            train_df.loc[len(train_df.index)] = ['rotated-'+file, train_df[train_df['image_title']==file]['class_label'].iloc[0]] 
    return train_df

