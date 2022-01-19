import torch
import torch.optim as optim
import torch.autograd.functional as AF
import math
from pytorch3d.ops import cot_laplacian

# An optimizer for meshes, which takes a parameter that returns some measure of energy on the
# mesh.
class UniformAdam(optim.Optimizer):
    """
    `UniformAdam` implements the optimizer from the paper "Large Steps in Inverse Rendering".

    The optimizer only works over meshes, and acts similarly to the standard Adam Optimizer, but
    additionally computes a Laplacian Matrix of the mesh to smooth the optimization process.
    """

    def __init__(
        self,
        params,
        mesh_from_params,
        lr: float = 5e-3,
        betas=(0.9, 0.999),
        diffusion: float = 1e-2,
        eps: float = 1e-5,
        method="cot",
    ):
        """
        Args:
            params: optimization parameters, 1-1 correspondece
                per vertex of the constructed mesh.
            mesh_from_params: F(params) -> mesh
            lr: learning rate.
            betas: momentum parameters, corresponding to those in the original adam paper.
            diffusion: laplacian multiplier, can vary based on laplacian.
            eps: value added to denominator for numerical stability.
            method: laplacian matrix construction method,
              supports either "uniform" or "cot".
        """
        assert lr > 0, "Must assign a learning rate greater than 0"
        assert (
            mesh_from_params is not None
        ), "Must pass a function that maps from params to mesh"
        defaults = dict(
            mesh_from_params=mesh_from_params,
            lr=lr,
            betas=betas,
            diffusion=diffusion,
            eps=eps,
        )

        super().__init__(params, defaults)
        if method == "uniform":
            self.method = lambda mesh: mesh.laplacian_packed()
        elif method == "cot":
            self.method = lambda mesh: cot_laplacian(mesh)[0]
        else:
            raise Exception(f"Unknown laplacian method: {method}")

    def step(self, closure=None):
        """
        Updates the parameters of this optimizer.

        Args:
          closure: An optional closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta_1, beta_2 = group["betas"]
            diff = group["diffusion"]
            mesh_from_params = group["mesh_from_params"]
            step_size = group["lr"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                dphidx = p.grad.data
                assert not dphidx.is_sparse, "Does not support sparse grads yet"
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["moving_avg_1"] = torch.zeros_like(p.grad)
                    state["moving_avg_2"] = torch.zeros_like(p.grad)
                    state["g"] = torch.zeros_like(p.data)
                    state["u"] = torch.zeros_like(p.data)
                state["step"] += 1

                g = state["g"]
                m_1, m_2 = state["moving_avg_1"], state["moving_avg_2"]

                mesh = mesh_from_params(p.data)

                L = self.method(mesh).mul_(diff)
                assert len(L.shape) == 2
                assert L.shape[0] == L.shape[1]
                L = torch.eye(L.shape[0], device=L.device) + L

                torch.linalg.solve(L, dphidx, out=g)
                m_1.mul_(beta_1).add_(g, alpha=(1 - beta_1))
                m_2.mul_(beta_2).addcmul_(g, g, value=(1 - beta_2))

                n_step = state["step"]

                bias_correction_1 = 1 - (beta_1 ** n_step)
                bias_correction_2 = 1 - (beta_2 ** n_step)
                step_size = step_size * math.sqrt(bias_correction_2) / bias_correction_1

                u = state["u"]
                torch.matmul(L, p.data, out=u)
                u.addcdiv_(
                    m_1,
                    torch.linalg.vector_norm(
                        m_2, ord=float("inf"), dim=-1, keepdim=True
                    )
                    .sqrt_()
                    .add_(eps),
                    value=-step_size,
                )
                torch.linalg.solve(L, u, out=p.data)
        return loss
