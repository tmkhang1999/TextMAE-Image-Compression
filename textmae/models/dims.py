"""
Channel plans for the compression layers, kept free of torch so they are easy to test.

Each plan interpolates between two widths in quarter steps, e.g.
`interp(768, 512, 3) -> 704`, `interp(768, 512, 2) -> 640`.
"""


def interp(start, end, quarters):
    """Width that is `quarters`/4 of the way from `end` towards `start`."""
    return int(end + (start - end) * quarters / 4)


def analysis_dims(encoder_dim, decoder_dim, latent_dim):
    """g_a: encoder tokens -> latent y."""
    return [
        encoder_dim,
        interp(encoder_dim, decoder_dim, 3),
        interp(encoder_dim, decoder_dim, 2),
        decoder_dim,
        latent_dim,
    ]


def synthesis_dims(encoder_dim, decoder_dim, latent_dim):
    """g_s: latent y_hat -> encoder-width tokens."""
    return [
        latent_dim,
        decoder_dim,
        interp(encoder_dim, decoder_dim, 2),
        interp(encoder_dim, decoder_dim, 3),
        encoder_dim,
    ]


def hyper_analysis_dims(latent_dim, hyper_dim):
    """h_a: latent y -> hyper-latent z (two stride-2 steps)."""
    return [
        latent_dim,
        latent_dim,
        interp(latent_dim, hyper_dim, 3),
        interp(latent_dim, hyper_dim, 2),
        interp(latent_dim, hyper_dim, 1),
        hyper_dim,
    ]


def hyper_synthesis_dims(latent_dim, hyper_dim):
    """h_s_mean / h_s_scale: z_hat -> per-latent statistics (two sub-pixel up-steps)."""
    return [
        hyper_dim,
        interp(latent_dim, hyper_dim, 1),
        interp(latent_dim, hyper_dim, 2),
        interp(latent_dim, hyper_dim, 3),
        latent_dim,
        latent_dim,
    ]


def channel_context_dims(latent_dim, num_slices, slice_index, in_extra=0):
    """
    Widths of one channel-context transform (cc_transform_mean/scale and lrp_transform).

    Input = hyper-prior statistics + the already decoded slices (at most `num_slices // 2`,
    `+ in_extra` for the LRP transform which also sees the current slice).
    Output = one slice of the latent.
    """
    slice_dim = latent_dim // num_slices
    half = num_slices // 2
    in_dim = int(latent_dim + slice_dim * min(slice_index + in_extra, half + in_extra))
    return [
        in_dim,
        int(slice_dim * (half + 1)),
        int(slice_dim * (half * 3 / 4 + 1)),
        int(slice_dim * (half * 2 / 4 + 1)),
        int(slice_dim * (half * 1 / 4 + 1)),
        int(slice_dim),
    ]
