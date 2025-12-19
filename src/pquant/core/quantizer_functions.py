def create_fixed_quantizer(k, i, f, overflow, round_mode):
    from quantizers import get_fixed_quantizer

    quantizer = get_fixed_quantizer(round_mode=round_mode, overflow_mode=overflow)
    return quantizer


def create_hgq_parameters_quantizer(k, i, f, overflow, round_mode):
    from hgq.quantizer import Quantizer

    return Quantizer(
        k0=k,
        i0=i,
        f0=f,
        round_mode=round_mode,
        overflow_mode=overflow,
        q_type="kif",
        homogeneous_axis=(),
    )


def create_hgq_data_quantizer(k, i, f, overflow, round_mode):
    from hgq.quantizer import Quantizer

    return Quantizer(
        k0=k,
        i0=i,
        f0=f,
        round_mode=round_mode,
        overflow_mode=overflow,
        q_type="kif",
        homogeneous_axis=(0,),
    )


def create_quantizer(k, i, f, overflow, round_mode, is_heterogeneous, is_data):
    if is_heterogeneous:
        if is_data:
            return create_hgq_data_quantizer(k, i, f, overflow, round_mode)
        else:
            return create_hgq_parameters_quantizer(k, i, f, overflow, round_mode)
    else:
        return create_fixed_quantizer(k, i, f, overflow, round_mode)
