# Creating Train / Val / Test folders (One time use)
import argparse

from yawn_train.src.split_data_into_datasets_manager import SplitDatasetManager

_HELP_TEXT = """
Usage Examples:
python -m tf2onnx.convert --saved-model saved_model_dir --output model.onnx
python -m tf2onnx.convert --input frozen_graph.pb  --inputs X:0 --outputs output:0 --output model.onnx
python -m tf2onnx.convert --checkpoint checkpoint.meta  --inputs X:0 --outputs output:0 --output model.onnx
For help and additional information see:
    https://github.com/onnx/tensorflow-onnx
If you run into issues, open an issue here:
    https://github.com/onnx/tensorflow-onnx/issues
"""


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser(description="Convert tensorflow graphs to ONNX.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_HELP_TEXT)
    parser.add_argument("--input_folder", type=str, default='/Users/igla/Downloads/mouth_state_new6-3/',
                        help="input_folder")
    parser.add_argument("--classes", type=str, default='opened, closed',
                        help="classes")
    parser.add_argument("--include_hidden", type=bool, default=False,
                        help="include hidden files")
    args = parser.parse_args()
    args.shape_override = None
    if args.classes:
        args.classes = args.classes.split(",")
    return args


if __name__ == '__main__':
    args = get_args()
    if args.input_folder:
        root_dir = args.input_folder
        classes_dir = args.classes

        split = SplitDatasetManager(root_dir, classes_dir)
        split.prepare()
