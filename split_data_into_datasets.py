# Creating Train / Val / Test folders (One time use)
from yawn_train.split_data_into_datasets_manager import SplitDatasetManager

root_dir = 'mouth_state/'
classes_dir = ['opened/', 'closed/']

split = SplitDatasetManager(root_dir, classes_dir)
split.prepare()
