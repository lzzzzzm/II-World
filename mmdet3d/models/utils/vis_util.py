import numpy as np
import cv2 as cv
import torch

color_map = np.array(
    [
        [0, 0, 0],          # unlabeled            black
        [255, 120, 50],     # barrier              orange
        [255, 192, 203],    # bicycle              pink
        [255, 255, 0],      # bus                  yellow
        [0, 150, 245],      # car                  blue
        [0, 255, 255],      # construction_vehicle cyan
        [255, 127, 0],      # motorcycle           dark orange
        [255, 0, 0],        # pedestrian           red
        [255, 240, 150],    # traffic_cone         light yellow
        [135, 60, 0],       # trailer              brown
        [160, 32, 240],     # truck                purple
        [255, 0, 255],      # driveable_surface    dark pink
        [139, 137, 137],    # other_flat           dark red
        [75, 0, 75],        # sidewalk             dard purple
        [150, 240, 80],     # terrain              light green
        [230, 230, 250],    # manmade              white
        [0, 175, 0],        # vegetation           green
        [255, 255, 255]     # free                 white
    ]
)

def change_occupancy_to_bev(occ_semantics, occ_size=(200, 200, 16), free_cls=17):
    # free_cls == 16 as default

    semantics_valid = np.logical_not(occ_semantics == free_cls)
    d = np.arange(occ_size[-1]).reshape(1, 1, occ_size[-1])
    d = np.repeat(d, occ_size[0], axis=0)
    d = np.repeat(d, occ_size[1], axis=1).astype(np.float32)
    d = d * semantics_valid
    selected = np.argmax(d, axis=2)
    selected_torch = torch.from_numpy(selected)
    semantics_torch = torch.from_numpy(occ_semantics)

    occ_bev_torch = torch.gather(semantics_torch, dim=2, index=selected_torch.unsqueeze(-1))
    occ_bev = occ_bev_torch.numpy()
    occ_bev = occ_bev.flatten().astype(np.int32)
    occ_bev_vis = color_map[occ_bev].astype(np.uint8)
    occ_bev_vis = occ_bev_vis.reshape(occ_size[0], occ_size[1], 3)[::-1, ::-1, :3]
    occ_bev_vis = cv.resize(occ_bev_vis, (occ_size[0], occ_size[1]))
    occ_bev_vis = cv.cvtColor(occ_bev_vis, cv.COLOR_RGB2BGR)

    return occ_bev_vis