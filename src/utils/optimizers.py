import torch.optim as optim


def configure_optimizers(net, args):
    """Main optimizer for everything except the entropy-bottleneck quantiles, auxiliary one for those."""
    params = dict(net.named_parameters())
    aux = sorted(n for n, p in params.items() if n.endswith(".quantiles") and p.requires_grad)
    main = sorted(n for n, p in params.items() if not n.endswith(".quantiles") and p.requires_grad)
    return (optim.Adam((params[n] for n in main), lr=args.learning_rate),
            optim.Adam((params[n] for n in aux), lr=args.aux_learning_rate))
