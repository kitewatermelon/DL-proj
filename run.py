from configs import parse_args, get_device
from data_utils import load_balanced_dataset
from trainer_runner import train_and_evaluate

if __name__ == "__main__":
    args = parse_args()
    device = get_device(args.cuda)

    print(f"🎯 선택한 버전: {args.version}, 디바이스: {device}")
    train_df, val_df, test_df = load_balanced_dataset(args.version)
    train_and_evaluate(args.version, train_df, val_df, test_df, device)
