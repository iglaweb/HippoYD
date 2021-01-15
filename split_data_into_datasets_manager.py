import os
import shutil

import numpy as np


class SplitDatasetManager(object):
    train_ratio = 0.80
    val_ratio = 0.15
    test_ratio = 0.05

    def __init__(self, root_dir, classes_dir):
        self.root_dir = root_dir
        self.classes_dir = classes_dir

    def prepare(self):
        for cls in self.classes_dir:
            print(f'Current class: {cls}')
            os.makedirs(self.root_dir + 'train/' + cls, exist_ok=True)
            os.makedirs(self.root_dir + 'val/' + cls, exist_ok=True)
            os.makedirs(self.root_dir + 'test/' + cls, exist_ok=True)

            # Creating partitions of the data after shuffeling
            src_folder = self.root_dir + cls  # Folder to copy images from

            allFileNames = os.listdir(src_folder)
            np.random.shuffle(allFileNames)
            train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                                      [int(len(allFileNames) * self.train_ratio),
                                                                       int(len(allFileNames) * (
                                                                               self.train_ratio + self.val_ratio))])

            train_FileNames = [src_folder + '/' + name for name in train_FileNames.tolist()]
            val_FileNames = [src_folder + '/' + name for name in val_FileNames.tolist()]
            test_FileNames = [src_folder + '/' + name for name in test_FileNames.tolist()]

            print('Total images: ', len(allFileNames))
            print('Training: ', len(train_FileNames))
            print('Validation: ', len(val_FileNames))
            print('Testing: ', len(test_FileNames))

            # Copy-pasting images
            for name in train_FileNames:
                dst = os.path.join(self.root_dir, 'train/', cls)
                shutil.copy(name, dst)

            for name in val_FileNames:
                dst = os.path.join(self.root_dir, 'val/', cls)
                shutil.copy(name, dst)

            for name in test_FileNames:
                dst = os.path.join(self.root_dir, 'test/', cls)
                shutil.copy(name, dst)

        print('Ready!')
