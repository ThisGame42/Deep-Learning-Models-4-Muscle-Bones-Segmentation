import argparse

from DeepLearning.Training.Preparation import start_training, start_testing


def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ('yes', 'true', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser(description="Supply parameters to train the model.")
    parser.add_argument("--training_img_path", type=str, nargs='?', const=True, default=False,
                        help="The absolute path to the folder where the training images can be found.")
    parser.add_argument("--training_label_path", type=str, nargs='?', const=True, default=False,
                        help="The absolute path to the folder where the training labels can be found.")
    parser.add_argument("--model_name", type=str, nargs='?', const=True, default=False,
                        help="The name of the model.")
    parser.add_argument("--val_img_path", type=str, nargs='?', const=True, default=5,
                        help="The absolute path to the folder where the validation images can be found.")
    parser.add_argument("--val_label_path", type=str, nargs='?', const=True, default=False,
                        help="The absolute path to the folder where the validation labels can be found.")
    parser.add_argument("--loss_fn", type=str, nargs='?', const=True, default=False,
                        help="The name of the loss function.")
    parser.add_argument("--optimiser", type=str, nargs='?', const=True, default=False,
                        help="The name of the optimiser.")
    parser.add_argument("--num_epochs", type=int, nargs='?', const=True, default=False,
                        help="The number of epochs to train the model for.")
    parser.add_argument("--output_path", type=str, nargs='?', const=True, default=False,
                        help="The location where the weights of the model will be saved.")
    parser.add_argument("--batch_size", type=int, nargs='?', const=True, default=False,
                        help="The batch size.")
    parser.add_argument("--num_classes", type=int, nargs='?', const=True, default=False,
                        help="The number of muscles in the training images.")
    parser.add_argument("--learning_rate", type=float, nargs='?', const=True, default=0.001,
                        help="The initial learning rate.")
    parser.add_argument("--pre_training", type=str2bool, nargs='?', const=True, default=False,
                        help="Whether to use pre-trained weights or not.")
    parser.add_argument("--is_2d", type=str2bool, nargs='?', const=True, default=False,
                        help="True if training a 2D model, and false otherwise.")
    parser.add_argument("--num_slices", type=int, nargs='?', const=True, default=16,
                        help="The number of slices in each sub-volume.")
    parser.add_argument("--fine_tuning", type=str2bool, nargs='?', const=True, default=False,
                        help="Whether to fine-tune a model or not.")
    parser.add_argument("--model_path", type=str, nargs='?', const=True, default="",
                        help="The path to the model to be fine-tuned.")
    parser.add_argument("--test", type=str2bool, nargs='?', const=True, default=False,
                        help="Whether to test a trained model or not.")
    parser.add_argument("--testing_path", type=str, nargs='?', const=True, default="",
                        help="The path to the test data.")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if not args.test:
        start_training(args)
    else:
        start_testing(args)
