from yawn_train.src.rebalance_classes_manager import RebalanceManager

if __name__ == '__main__':
    classes = ['opened', 'closed']
    src_folder = '/Users/igla/Downloads/mouth_state_new10-5'
    dst_folder = '/Users/igla/Downloads/mouth_state_new10-5_rebalanced'
    split_dm = RebalanceManager(src_folder, classes, dst_folder)
    split_dm.prepare()
