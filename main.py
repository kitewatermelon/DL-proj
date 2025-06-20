import argparse
from dataset import load_all_datasets
from train import train_model
from test import test_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='dataset name or "all"')
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--valid_ratio', type=float, default=0.05)
    parser.add_argument('--test_ratio', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help="train or test mode")
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    total = args.train_ratio + args.valid_ratio + args.test_ratio
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"[❌] 합이 1이 되도록 비율을 설정하세요. 현재 합계: {total}")

    if args.dataset.lower() == 'all':
        datasets = ['turingbench', 'aivshuman', 'daigt', 'gptwriting', 'gpt2output']
    else:
        datasets = [args.dataset]

    train_df, valid_df, test_df = load_all_datasets(
        datasets,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio
    )

    print(f"[✅] Train: {len(train_df):,} / Valid: {len(valid_df):,} / Test: {len(test_df):,}")

    if args.mode == 'train':
        train_model(train_df, valid_df, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
    elif args.mode == 'test':
        test_model(test_df, batch_size=args.batch_size, device=args.device)

if __name__ == '__main__':
    main()
