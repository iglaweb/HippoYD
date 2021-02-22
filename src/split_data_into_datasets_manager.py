import os
import shutil

import numpy as np


class SplitDatasetManager(object):
    train_ratio = 0.80
    val_ratio = 0.15
    test_ratio = 0.05

    def __init__(self, root_dir, classes_dir, include_hidden_files=False):
        self.root_dir = root_dir
        self.classes_dir = classes_dir
        self.include_hidden_files = include_hidden_files

    def listdir_nohidden(self, path):
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f

    def prepare(self):
        for cls in self.classes_dir:
            print(f'Current class: {cls}')
            os.makedirs(os.path.join(self.root_dir, 'train/', cls), exist_ok=True)
            os.makedirs(os.path.join(self.root_dir, 'val/', cls), exist_ok=True)
            os.makedirs(os.path.join(self.root_dir, 'test/', cls), exist_ok=True)

            # Creating partitions of the data after shuffeling
            src_folder = os.path.join(self.root_dir, cls)  # Folder to copy images from

            all_file_names = list(self.listdir_nohidden(src_folder))
            np.random.shuffle(all_file_names)
            train_file_names, val_file_names, test_file_names = np.split(np.array(all_file_names),
                                                                         [int(len(all_file_names) * self.train_ratio),
                                                                          int(len(all_file_names) * (
                                                                                  self.train_ratio + self.val_ratio))])

            train_file_names = [os.path.join(src_folder, name) for name in train_file_names.tolist()]
            val_file_names = [os.path.join(src_folder, name) for name in val_file_names.tolist()]
            test_file_names = [os.path.join(src_folder, name) for name in test_file_names.tolist()]

            print('Total images: ', len(all_file_names))
            print('Training: ', len(train_file_names))
            print('Validation: ', len(val_file_names))
            print('Testing: ', len(test_file_names))

            # Copy-pasting images
            for name in train_file_names:
                dst = os.path.join(self.root_dir, 'train/', cls)
                shutil.copy(name, dst)

            for name in val_file_names:
                dst = os.path.join(self.root_dir, 'val/', cls)
                shutil.copy(name, dst)

            for name in test_file_names:
                dst = os.path.join(self.root_dir, 'test/', cls)
                shutil.copy(name, dst)

        print('Ready!')
