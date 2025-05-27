import argparse
from dataset import load_all_datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of train set (default: 0.7)')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='Ratio of validation set (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of test set (default: 0.15)')
    args = parser.parse_args()

    # 비율 검증
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

if __name__ == '__main__':
    main()
