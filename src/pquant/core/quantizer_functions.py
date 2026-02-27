import keras


def create_fixed_quantizer(k, i, f, overflow, round_mode):
    if keras.backend.backend() == "torch":
        from pquant.core.torch.fixed_point_quantizer import get_fixed_quantizer
    else:
        from quantizers import get_fixed_quantizer

    quantizer = get_fixed_quantizer(round_mode=round_mode, overflow_mode=overflow)
    return quantizer


def create_hgq_parameters_quantizer(k, i, f, overflow, round_mode, place):
    from hgq.quantizer import Quantizer, QuantizerConfig

    quantizer_config = QuantizerConfig(
        q_type="kif", place=place, k0=k, i0=i, f0=f, overflow_mode=overflow, round_mode=round_mode, homogeneous_axis=()
    )

    return Quantizer(config=quantizer_config)


def create_hgq_data_quantizer(k, i, f, overflow, round_mode):
    from hgq.quantizer import Quantizer, QuantizerConfig

    quantizer_config = QuantizerConfig(
        q_type="kif",
        place="datalane",
        k0=k,
        i0=i,
        f0=f,
        overflow_mode=overflow,
        round_mode=round_mode,
        homogeneous_axis=(0,),
    )

    return Quantizer(config=quantizer_config)


def create_quantizer(k, i, f, overflow, round_mode, is_heterogeneous, is_data, place="datalane"):
    if is_heterogeneous:
        if is_data:
            return create_hgq_data_quantizer(k, i, f, overflow, round_mode)
        else:
            return create_hgq_parameters_quantizer(k, i, f, overflow, round_mode, place)
    else:
        return create_fixed_quantizer(k, i, f, overflow, round_mode)
