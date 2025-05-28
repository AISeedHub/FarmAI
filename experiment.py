import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
from utils.config_loader import load_config


def main():
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Transformer models based family for Time Series Forecasting')

    # config files
    parser.add_argument('--config', type=str, default=None,
                        help='model config file path (e.g., tcn/default_config). If provided, other arguments will be ignored except for --is_training and --experiment')
    parser.add_argument('--experiment', type=str, default='default',
                        help='experiment config name (e.g., default, multivariate). Defaults to "default"')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--is_pretrain', type=int, required=False, default=0, help='status')
    parser.add_argument('--is_finetune', type=int, required=False, default=0, help='status')
    parser.add_argument('--pretrained_model', type=str, required=False, default=None, help='pretrained model path')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=False, default='Transformer',
                        help='model name, options: [Transformer, Informer, Autoformer, PatchTST, LSPatchT, iTransformer, Crossformer]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='ETTh1',
                        help='dataset type for choosing Data Loader')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./experiments/model_saved/checkpoints/',
                        help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; '
                             'M:multivariate predict multivariate, '
                             'S:univariate predict univariate, '
                             'MS:multivariate predict univariate')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    # parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    # parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio for the input')
    parser.add_argument('--patch_len', type=int, default=12, help='patch length')
    parser.add_argument('--stride', type=int, default=12, help='stride between patch')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

    # Parse command line arguments
    args = parser.parse_args()

    # Load configuration from file if provided
    if args.config:
        try:
            # Extract config name from config path (e.g., 'tcn/default_config' -> 'default_config')
            config_name = args.config
            # Get experiment name from command line arguments
            experiment_name = args.experiment

            # Load the configuration
            config_args = load_config(config_name, experiment_name)

            # Keep the is_training flag from command line arguments
            is_training = args.is_training

            # override args with loaded config
            # Lặp qua các thuộc tính của đối tượng SimpleNamespace
            for key in dir(config_args):
                # Bỏ qua các thuộc tính dạng __xxx__
                if not key.startswith('__') and not key.endswith('__'):
                    value = getattr(config_args, key)
                    if hasattr(args, key):
                        setattr(args, key, value)

            # Restore the is_training flag
            args.is_training = is_training

            print(f"Model configuration loaded from: config/model/{config_name}.json")
            print(f"Experiment configuration loaded from: config/experiment/{experiment_name}.json")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using command line arguments instead.")

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.is_training:
        # setting record of experiments
        setting = get_exp_setting(args)
        if args.is_pretrain:
            args.itr = 1
        elif args.is_finetune:  # fine tuning or supervised
            if args.pretrained_model is None:
                raise ValueError("Please provide the path of the pretrained model")
            if not os.path.exists(args.pretrained_model):
                print('>>>>>>>supervised : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

        best_model_path = None
        for ii in range(args.itr):
            setting_name = f'{setting}_{ii}'
            exp = Exp_Main(args, setting=setting_name)
            best_model_path = run_experiment(exp, setting_name, args)

        if best_model_path:
            print('>>>>>>>best model path : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(best_model_path))
            if args.is_finetune and args.is_pretrain:  # fine tuning
                args.itr = 2
                args.is_pretrain = 0
                args.batch_size = 32
                args.train_epochs = 10
                args.patch_len = args.seq_len
                args.stride = args.seq_len
                args.model_id = "FineTune"
                args.pretrained_model = best_model_path
                setting = get_exp_setting(args)
                print('>>>>>>>fine tuning : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                print("Loading pretrained model from: ", best_model_path)

                for ii in range(args.itr):
                    setting_name = f'{setting}_{ii}'
                    exp = Exp_Main(args, setting=setting_name)
                    run_experiment(exp, setting_name, args)
    else:
        setting = f'{get_exp_setting(args)}_{0}'
        exp = Exp_Main(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()


def get_exp_setting(args):
    if args.is_pretrain:
        return '{}_{}_{}_sl{}'.format(
            args.model_id, args.model, args.data, args.seq_len)  # set the name of pretrain model
    return '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
        args.model_id, args.model, args.data, args.features, args.seq_len, args.label_len,
        args.pred_len, args.d_model, args.n_heads, args.e_layers, args.d_layers,
        args.d_ff, args.factor, args.embed, args.distil, args.des)


def run_experiment(exp, setting_name, args):
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting_name))
    best_model_path = exp.train(setting_name)
    if args.is_pretrain:
        return best_model_path

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting_name))
    exp.test(setting_name, best_model_path)

    torch.cuda.empty_cache()
    return best_model_path


if __name__ == "__main__":
    main()
