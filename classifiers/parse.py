from models import *
from torchvision import models

height, width = 28, 28
n_channels = 1

def parse_method(args, device, dataset):
    n_classes = len(dataset.encoding)
    if args.method == 'mlp':
        in_channels = height*width*n_channels
        model = MLP(in_channels=in_channels,
                    hidden_channels=args.hidden_channels, 
                    out_channels=n_classes,
                    num_layers=args.num_layers)
    elif args.method == 'resnet':
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
    elif args.method == 'cnn':
        n_classes = 11
        in_channels=height*width*n_channels
        model = CNN(in_channels=in_channels,
                    hidden_channels = 32,
                    out_channels=n_classes,
                    kernel_size=3,
                    dropout=.5)
    return model

def parser_add_main_args(parser):
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--method', '-m', type=str, default='mlp')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--num_workers', type=int,
                        default=0, help='number of workers to use for dataloader')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--hops', type=int, default=1,
                        help='power of adjacency matrix for certain methods')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--adam', action='store_true', help='use adam instead of adamW')
    parser.add_argument("--SGD", action='store_true', help='Use SGD as optimizer')

