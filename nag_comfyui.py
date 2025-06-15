import torch


class NagCore:
    def __init__(self, tau=2.5, alpha=0.5, p=1):
        self.tau = tau
        self.alpha = alpha
        self.p = p

    def nag(self, cond, uncond, cfg_scale):
        """
        Normalized Attention Guidance (NAG) implementation.

        Args:
            cond (torch.Tensor): Conditional tensor (positive prompt) of shape (batch, 4, height, width).
            uncond (torch.Tensor): Unconditional tensor (negative prompt) of shape (batch, 4, height, width).
            cfg_scale (float): Classifier-free guidance scale.
        """
        assert cond.shape[0] == uncond.shape[0], "Batch sizes of cond and uncond must match"
        z_pos = cond
        z_neg = uncond

        z_tilde = z_pos + cfg_scale * (z_pos - z_neg)

        # Calculate the norm ratio
        norm_pos = torch.norm(z_pos, p=self.p, dim=(-2, -1), keepdim=True)
        norm_tilde = torch.norm(z_tilde, p=self.p, dim=(-2, -1), keepdim=True)
        ratio = norm_tilde / norm_pos

        # Apply the tau threshold
        tau = torch.tensor(self.tau, dtype=ratio.dtype, device=ratio.device)
        z_hat = torch.where(ratio > tau, tau, ratio) / ratio * z_tilde

        # Apply the alpha blending
        z_nag = self.alpha * z_hat + (1 - self.alpha) * z_pos

        return z_nag


class NAGLatent:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "tau": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                "alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/unet"

    def patch(self, model, tau, alpha):
        core = NagCore(tau, alpha)

        def sampler_nag(args):
            input = args["input"]
            cond = input - args["cond"]
            uncond = input - args["uncond"]
            cond_scale = args["cond_scale"]
            return input - core.nag(cond, uncond, cond_scale)

        m = model.clone()
        m.set_model_sampler_cfg_function(sampler_nag)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "NAG Latent": NAGLatent,
}