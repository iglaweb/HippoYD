import os
import shutil

import numpy as np


class RebalanceManager(object):

    def __init__(self, root_dir, classes_dir, dst_folder):
        self.root_dir = root_dir
        self.classes_dir = classes_dir
        self.dst_folder = dst_folder

    def listdir_nohidden(self, path):
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f

    def get_smalles_cnt(self):
        min_files = -1
        for cls in self.classes_dir:
            src_folder = os.path.join(self.root_dir, cls)

            list = os.listdir(src_folder)  # dir is your directory path
            number_files = len(list)
            if min_files == -1 or number_files < min_files:
                min_files = number_files
        return min_files

    def rebalance(self, all_file_names, dst_folder_cls, count_files):
        all_file_names = all_file_names[:count_files]
        # Copy-pasting images
        for file in all_file_names:
            filename = os.path.basename(file)
            dst = os.path.join(dst_folder_cls, filename)
            shutil.copy(file, dst)

    def prepare(self):
        min_files = self.get_smalles_cnt()
        if min_files == 0:
            print('Some class does not have images at all')
            return
        for cls in self.classes_dir:
            print(f'Current class: {cls}')

            # Creating partitions of the data after shuffeling
            dst_folder_cls = os.path.join(self.dst_folder, cls)
            os.makedirs(dst_folder_cls, exist_ok=True)

            src_folder = os.path.join(self.root_dir, cls)  # Folder to copy images from
            all_file_names = list(self.listdir_nohidden(src_folder))
            np.random.shuffle(all_file_names)

            all_file_paths = [os.path.join(src_folder, name) for name in all_file_names]
            self.rebalance(all_file_paths, dst_folder_cls, min_files)
        print('Ready!')
