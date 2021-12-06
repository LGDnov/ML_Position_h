# working with data set
import os
import pandas as pd
import shutil

def create_file_system():
    # In Source folder: df.csv file and img folder
    train = pd.read_csv('Source/df.csv')
    train = train.set_index('Unnamed: 0')
    name_group = train.positions.unique()
    #creating a file system for training the model
    os.mkdir("Source/Data_set")
    os.mkdir("Source/Data_set/Valid")
    os.mkdir("Source/Data_set/Train")
    for i in range(len(name_group)):
       os.mkdir("Source/Data_set/Train"+name_group[i])
    for i in range(len(name_group)):
       os.mkdir("Source/Data_set/Valid"+name_group[i])
    print(len(train.positions))
    for i in range(len(train.positions)):
       if i>=35000:
          shutil.copy2('Source/'+str(train.image_name[i]), 'Source/Data_set/Valid/'+str(train.positions[i])+'/'+str(i)+'.jpg')
       else:
          shutil.copy2('Source/'+str(train.image_name[i]), 'Source/Data_set/Train/'+str(train.positions[i])+'/'+str(i)+'.jpg')

if __name__=='__main__':
    create_file_system()