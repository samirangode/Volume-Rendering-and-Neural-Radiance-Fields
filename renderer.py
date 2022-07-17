import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):
        # TODO (1.5): Compute transmittance using the equation described in the README
        #T = torch.ones((deltas.shape[0], 1)).cuda()
        #weights = torch.ones_like(deltas)
        #trans = torch.exp(-rays_density * deltas + eps)
        # for i in range(1, T.shape[1]):
        # T[:, i] = T[:, i-1] * \
        ##        torch.exp(-rays_density[:, i-1] * deltas[:, i-1] + eps)

        # for i in range(1, deltas.shape[1]):
        #    print((T[i-1, :] * torch.exp(-rays_density[:, i-1]
        #          * deltas[:, i-1] + eps)).shape)
        #    T = torch.vstack(
        #        (T, T[i-1, :] * torch.exp(-rays_density[:, i-1] * deltas[:, i-1] + eps)))
        #T = T.view(-1, deltas.shape[1], 1)
        ## print(T[30000:30004, ...])
        # print(deltas)

        # TODO (1.5): Compute weight used for rendering from transmittance and density
        #weights = T * (1 - trans)

        N, d = deltas.size(0), deltas.size(1)  # deltas shape: (4096, 64, 1)

        T = []
        T_prev = torch.ones((N, 1)).cuda()

        T.append(T_prev)

        for i in range(0, d - 1):

            T_curr = T_prev * \
                torch.exp(- rays_density[:, i] * deltas[:, i] + eps)
            T_prev = T_curr

            T.append(T_curr)

        T = torch.stack(T, dim=1)

        # TODO (1.5): Compute weight used for rendering from transmittance and density

        weights = T * (1. - torch.exp(- rays_density * deltas + eps))

        return weights

    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        # print(rays_feature.shape, weights.shape, "stuff")
        # print("OH YEAH OOOOOOOOOOOOOH YEAH H YEAH H YEAH H YEAH H YEAH H YEAH H YEAH H YEAH H YEAH H YEAH H YEAH H YEAH H YEAH ")
        #feature = (weights * rays_feature).sum(dim=1)
        # print(feature.shape, "feature")
        N, d = weights.size(0), weights.size(1)

        # print(weights.shape)
        # print(rays_feature.shape)

        feature = torch.sum(weights * rays_feature.view(N, d, -1), dim=1)

        return feature

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start +
                                        self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            density = implicit_output['density']
            feature = implicit_output['feature']

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            )

            # TODO (1.5): Render (color) features using weights
            #feature = self._aggregate(weights, feature.view(-1, 64, 3))
            feature = self._aggregate(weights, feature)

            # TODO (1.5): Render depth map
            #depth = self._aggregate(weights.squeeze(2), depth_values)
            depth = self._aggregate(weights, depth_values)

            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
                [chunk_out[k] for chunk_out in chunk_outputs],
                dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer
}
