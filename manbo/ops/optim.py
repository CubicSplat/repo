import tilelang
import tilelang.language as T
import torch

DEFAULT_THREADS = 256


@tilelang.jit
def adan_step_from_grad_kernel(
    threads: int = DEFAULT_THREADS,
    dtype: str = "float32",
    no_prox: bool = False,
    numel=T.dynamic("numel"),
):
    """
    TileLang Adan step kernel.

    Spec:
    - Inputs/outputs are flattened 1D tensors with same `numel`.
    - Dtype is float32 for params/states/grad.
    - Safe for `numel` not divisible by `threads`.
    - Updates param and optimizer states in-place.
    """

    if threads <= 0:
        raise ValueError("threads must be > 0")

    @T.prim_func
    def kernel(
        param: T.Tensor[[numel], dtype],
        grad: T.Tensor[[numel], dtype],
        exp_avg: T.Tensor[[numel], dtype],
        exp_avg_sq: T.Tensor[[numel], dtype],
        exp_avg_diff: T.Tensor[[numel], dtype],
        neg_pre_grad: T.Tensor[[numel], dtype],
        step_i32: T.int32,
        beta1: T.float32,
        beta2: T.float32,
        beta3: T.float32,
        bias_correction1: T.float32,
        bias_correction2: T.float32,
        bias_correction3_sqrt: T.float32,
        lr: T.float32,
        weight_decay: T.float32,
        eps: T.float32,
        clip_global_grad_norm: T.float32,
    ):
        one = T.Cast(dtype, 1.0)
        zero = T.Cast(dtype, 0.0)
        beta1_v = T.Cast(dtype, beta1)
        beta2_v = T.Cast(dtype, beta2)
        beta3_v = T.Cast(dtype, beta3)
        bias_correction1_v = T.Cast(dtype, bias_correction1)
        bias_correction2_v = T.Cast(dtype, bias_correction2)
        bias_correction3_sqrt_v = T.Cast(dtype, bias_correction3_sqrt)
        lr_v = T.Cast(dtype, lr)
        weight_decay_v = T.Cast(dtype, weight_decay)
        eps_v = T.Cast(dtype, eps)
        clip_global_grad_norm_v = T.Cast(dtype, clip_global_grad_norm)
        T.assume(step_i32 >= T.int32(1))

        grid = T.ceildiv(numel, threads)
        with T.Kernel(grid, threads=threads) as bx:
            T.annotate_safe_value(
                {
                    param: T.Cast(dtype, 0.0),
                    grad: T.Cast(dtype, 0.0),
                    exp_avg: T.Cast(dtype, 0.0),
                    exp_avg_sq: T.Cast(dtype, 0.0),
                    exp_avg_diff: T.Cast(dtype, 0.0),
                    neg_pre_grad: T.Cast(dtype, 0.0),
                }
            )
            for tx in T.Parallel(threads):
                idx = bx * threads + tx
                T.assume(idx >= 0)
                if idx < numel:
                    p = param[idx]
                    g = grad[idx] * clip_global_grad_norm_v

                    ngod = T.alloc_var(dtype)
                    if step_i32 == T.Cast(T.int32, 1):
                        ngod = zero
                    else:
                        ngod = neg_pre_grad[idx] + g

                    m_t = exp_avg[idx] * beta1_v + g * (one - beta1_v)
                    d_t = exp_avg_diff[idx] * beta2_v + ngod * (one - beta2_v)
                    ngod2 = ngod * beta2_v + g
                    n_t = exp_avg_sq[idx] * beta3_v + ngod2 * ngod2 * (one - beta3_v)

                    denom = T.sqrt(n_t) / bias_correction3_sqrt_v + eps_v
                    step_size = lr_v / bias_correction1_v
                    step_size_diff = lr_v * beta2_v / bias_correction2_v

                    p_new = T.alloc_var(dtype)
                    p_new = p
                    if no_prox:
                        p_new = p_new * (one - lr_v * weight_decay_v)
                        p_new = p_new - step_size * m_t / denom
                        p_new = p_new - step_size_diff * d_t / denom
                    else:
                        p_new = p_new - step_size * m_t / denom
                        p_new = p_new - step_size_diff * d_t / denom
                        p_new = p_new / (one + lr_v * weight_decay_v)

                    param[idx] = p_new
                    exp_avg[idx] = m_t
                    exp_avg_diff[idx] = d_t
                    exp_avg_sq[idx] = n_t
                    neg_pre_grad[idx] = -g

    return kernel


def adan_step_from_grad_tilelang(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    exp_avg_diff: torch.Tensor,
    neg_pre_grad: torch.Tensor,
    *,
    step: int,
    lr: float = 1e-3,
    betas: tuple[float, float, float] = (0.98, 0.92, 0.99),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    no_prox: bool = False,
    clip_global_grad_norm: float = 1.0,
    threads: int = DEFAULT_THREADS,
) -> None:
    """
    Apply one Adan step in-place from an externally provided gradient tensor.
    """

    beta1, beta2, beta3 = betas
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    bias_correction3 = 1.0 - beta3**step

    numel = param.numel()
    param_flat = param.view(numel)
    grad_flat = grad if grad.is_contiguous() else grad.contiguous()
    grad_flat = grad_flat.view(numel)
    exp_avg_flat = exp_avg.view(numel)
    exp_avg_sq_flat = exp_avg_sq.view(numel)
    exp_avg_diff_flat = exp_avg_diff.view(numel)
    neg_pre_grad_flat = neg_pre_grad.view(numel)

    kernel = adan_step_from_grad_kernel(
        threads=threads,
        dtype="float32",
        no_prox=bool(no_prox),
    )
    kernel(
        param_flat,
        grad_flat,
        exp_avg_flat,
        exp_avg_sq_flat,
        exp_avg_diff_flat,
        neg_pre_grad_flat,
        int(step),
        float(beta1),
        float(beta2),
        float(beta3),
        float(bias_correction1),
        float(bias_correction2),
        float(bias_correction3**0.5),
        float(lr),
        float(weight_decay),
        float(eps),
        float(clip_global_grad_norm),
    )
