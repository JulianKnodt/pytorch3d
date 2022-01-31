import os
import torch
import torch.nn.functional as F
import pytorch3d
from torchvision.utils import save_image

from pytorch3d.loss import ( mesh_optimizer )
import matplotlib.pyplot as plt

from pytorch3d.utils import ico_sphere
import numpy as np
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    SoftSilhouetteShader,
    HardPhongShader,
    TexturesVertex
)

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

torch.autograd.set_detect_anomaly(True)
SIZE=128
DATA_DIR = "./data"
obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")

mesh = load_objs_as_meshes([obj_filename], device=device)
verts = mesh.verts_packed()
center = verts.mean(dim=0)
scale = max((verts - center).abs().max(dim=0)[0])
mesh.offset_verts(-center)
mesh.scale_verts_(1/scale.item())

num_views = 20
elev = torch.linspace(0, 360, num_views, device=device)
azim = torch.linspace(-180, 180, num_views, device=device)
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
camera = OpenGLPerspectiveCameras(device=device, R=R[:1, ...], T=T[:1, ...])
raster_settings = RasterizationSettings(
    image_size=SIZE, blur_radius=0.0, faces_per_pixel=1,
)
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
    shader=HardPhongShader(device=device, cameras=camera, lights=lights)
)
meshes = mesh.extend(num_views)

# Render the cow mesh from each viewing angle
expected = renderer(meshes, cameras=cameras, lights=lights)

target_cameras = [OpenGLPerspectiveCameras(device=device, R=R[None, i, ...], T=T[None, i, ...]) for i in range(num_views)]
sigma = 1e-4
raster_settings_soft = RasterizationSettings(image_size=SIZE, blur_radius=np.log(1. / 1e-4 - 1.)*sigma, faces_per_pixel=10)

# Differentiable soft renderer using per vertex RGB colors for texture
renderer_textured = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings_soft),
    shader=HardPhongShader(device=device, cameras=camera, lights=lights)
)

src_mesh = ico_sphere(3, device)
verts_shape = src_mesh.verts_packed().shape
sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=True)

opt, mk_mesh = mesh_optimizer(src_mesh, lr=1e-3, laplacian_weight=19)
rgb_opt = torch.optim.Adam([sphere_verts_rgb], lr=1e-3)
epochs = 1000
views_per_iter = 5
for i in range(epochs):
  opt.zero_grad()
  mesh = mk_mesh()
  mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb)

  idxs = np.random.permutation(num_views).tolist()[:views_per_iter]
  got = renderer_textured(mesh.extend(views_per_iter), cameras=cameras[idxs], lights=lights)
  if i == 0: save_image(got.permute(0, 3, 1, 2), "init.png")

  exp = expected[idxs]
  loss = F.l1_loss(got, exp)
  print(loss)

  loss.backward()
  opt.step()
  rgb_opt.step()
  assert(mesh.verts_packed().isfinite().all())
save_image(got.permute(0, 3, 1, 2), "got.png")
save_image(exp.permute(0, 3, 1, 2), "exp.png")
