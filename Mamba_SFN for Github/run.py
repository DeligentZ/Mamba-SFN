import argparse
from utils.en_train import EnConfig, EnRun
from utils.ch_train import ChConfig, ChRun


def main(args):
    if args.dataset not in ['simi', 'eatd']:

        EnRun(EnConfig(batch_size=args.batch_size,learning_rate=args.lr,seed=args.seed, model=args.model,
                                    fuse_version=args.fuse_version, dataset_name=args.dataset,num_hidden_layers=args.num_hidden_layers))
    else:
        ChRun(ChConfig(batch_size=args.batch_size,learning_rate=args.lr,seed=args.seed, model=args.model,
                                    fuse_version=args.fuse_version, num_hidden_layers=args.num_hidden_layers))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size, mosi:16')
    parser.add_argument('--lr', type=float, default=6.5e-6, help='learning rate, recommended: 8e-6 for mosi, mosei, 1e-5 for sims')
    parser.add_argument('--model', type=str, default='mamba', help='concatenate(cc), Single-layer Augmented Gated Encoder(SAGE), mamba')
    parser.add_argument('--fuse_version', type=str, default='v2', help='version')
    parser.add_argument('--dataset', type=str, default='mosi', help='dataset name: mosi, mosei, simi, etda')
    parser.add_argument('--num_hidden_layers', type=int, default=1, help='number of hidden layers for cross-modality encoder')
    args = parser.parse_args()
    main(args)




