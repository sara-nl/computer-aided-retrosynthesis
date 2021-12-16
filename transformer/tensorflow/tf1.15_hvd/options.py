import os
import argparse
import time


def get_options():
    parser = argparse.ArgumentParser(description='Transformer retrosynthesis model.')

    # Model arguments
    parser.add_argument('--layers', type=int, default=3, help='Number of layers in encoder\'s module. Default 3.')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads. Default 8.')
    parser.add_argument('--n_hidden', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--embedding_size', type=int, default=64, help='Size of the embedding')
    parser.add_argument('--key_size', type=int, default=64, help='Key size')

    # Training options
    parser.add_argument('--epochs', type=int, default=1000, help='Number of options to train for')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--horovod', action='store_true', default=False, help='Run with horovod')
    parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda')
    parser.add_argument('--seed', type=int, default=0)

    # Learning rate scheduling arguments
    parser.add_argument('--warmup', type=int, default=16000, help='Number of warmup steps')
    parser.add_argument('--lr_factor', type=float, default=20, help='Learning rate multiplication factor')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd', 'adamax'], default='adam')
    parser.add_argument('--eps', type=float, default=1e-7, help='Epsilon for the Adam optimizer')
    parser.add_argument('--epochs_per_cycle', type=int, default=100, help='Epochs per cycle')
    parser.add_argument('--warmup_on_cyle', action='store_true', help='Warmup upon entering a new cycle')
    parser.add_argument('--exp_range_schedule', action='store_true', help='Use exponential range LR schedule')

    # Logging options
    parser.add_argument('--epochs_to_save', nargs='+', type=int, default=[600, 700, 800, 900, 999],
                        help='When to save model')
    parser.add_argument('--run_name', type=str, default='run', help='Where to save tensorboard logs')
    parser.add_argument('--output_path', type=str, default='output')

    # Running options
    parser.add_argument('--validate', action='store', type=str, help='Validation regime.', required=False)
    parser.add_argument('--predict', action='store', type=str, help='File to predict.', required=False)
    parser.add_argument('--train', action='store', type=str, help='File to train.', required=False)

    # Inference options
    parser.add_argument('--model', type=str, default='model.h5', help='A model to be used during validation')
    parser.add_argument('--temperature', type=float, default=1.3, help='Temperature for decoding')
    parser.add_argument('--beam', type=int, default=5, help='Beams size, must be 1 meaning greedy search or >= 5.')
    parser.add_argument('--max_predict', type=int, default=500, help='Max prediction length')

    opts = parser.parse_args()
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.output_dir = os.path.join(opts.output_path, opts.run_name)
    os.makedirs(opts.output_dir, exist_ok=True)
    opts.cuda = not opts.no_cuda

    return opts
