import torch
import torch.optim as optim
import torch.autograd.functional as AF
import math
from pytorch3d.ops import cot_laplacian
from pytorch3d.structures import Meshes

def mesh_optimizer(
  mesh,
  lr:float,
  laplacian_weight:float=19,
  betas=(0.9, 0.999),
  eps: float = 1e-8,
  method="uniform",
  alpha=None,
):
  """
  `mesh_optimizer` returns an optimizer over mesh parameters, as well a function that
    reconstructs the
            method: laplacian matrix construction method, supports either "uniform" or "cot".
  """
  verts = mesh.verts_packed()
  def change_of_variable_transform():
    if method == "cot":
      L = cot_laplacian(verts, mesh.faces_packed())[0].detach()
    elif method == "uniform":
      L = mesh.laplacian_packed()
    else: raise NotImplementedError()
    device=mesh.device
    V = verts.shape[0]
    idx = torch.arange(V, dtype=torch.long, device=device)
    I = torch.sparse_coo_tensor(
      torch.stack([idx, idx]),
      torch.ones(V, dtype=torch.float, device=mesh.device),
      (V,V)
    )
    if alpha is None:
      M = torch.add(I, laplacian_weight * L)
    else:
      assert(0 <= alpha and alpha < 1), "Alpha must be in the range [0, 1)"
      M = (1-alpha) * I + alpha * L
    return M.coalesce()
  M = change_of_variable_transform().detach()
  def to_vertex_space(M, u): return torch.linalg.solve(M.to_dense(), u)
  def from_vertex_space(M, v): return M @ v
  u = from_vertex_space(M, verts)
  u.requires_grad = True
  opt = UniformAdam([u],lr=lr,eps=eps,betas=betas)
  f = mesh.faces_packed()

  # assume there is no vertex normals since they would change as the mesh gets updated anyway.
  return opt, lambda:  Meshes(verts=to_vertex_space(M, u)[None], faces=f[None], textures=getattr(mesh, "textures", None))



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
        lr: float = 5e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-3,
    ):
        """
        Args:
            params: optimization parameters, 1-1 correspondece
                per vertex of the constructed mesh.
            lr: learning rate.
            betas: momentum parameters, corresponding to those in the original adam paper.
            eps: value added to denominator for numerical stability.
        """
        assert lr > 0, "Must assign a learning rate greater than 0"
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
        )

        super().__init__(params, defaults)
    def step(self, mesh=None, closure=None):
        """
        Updates the parameters of this optimizer.

        Args:
          closure: An optional closure that reevaluates the model and returns the loss.
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            beta_1, beta_2 = group["betas"]
            step_size = group["lr"]
            eps = group["eps"]

            with torch.no_grad():
              for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["moving_avg_1"] = torch.zeros_like(p.grad)
                    state["moving_avg_2"] = torch.zeros_like(p.grad)
                state["step"] += 1
                assert(grad.isfinite().all()), grad

                m_1, m_2 = state["moving_avg_1"], state["moving_avg_2"]

                m_1.mul_(beta_1).add_(grad, alpha=1 - beta_1)
                m_2.mul_(beta_2).addcmul_(grad, grad, value=1 - beta_2)

                n_step = state["step"]

                bias_correction_1 = 1 - (beta_1 ** n_step)
                bias_correction_2 = 1 - (beta_2 ** n_step)
                step_size = step_size * math.sqrt(bias_correction_2) / bias_correction_1

                denom = m_2.sqrt_().max().add_(eps)
                assert(denom.isfinite()), denom.item()
                p.data.addcdiv_(m_1, denom, value=-step_size)
        return loss
