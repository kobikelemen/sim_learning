import numpy as np
import open3d as o3d
import random
from scipy.spatial.transform import Rotation as R

import pybullet as pb
import pybullet_data
from sim import (
    SimGrasp,
    ObjectInfo,
    CUBE_SCALING,
)
import cv2
import matplotlib.pyplot as plt
import os

from placement_env import setup_placement_scenario

URDF_PATH = "/mnt/02D0BBD8D0BBCFE1/repos/gb_example/trossen_arm_description/urdf/generated/wxai/wxai_follower.urdf"
DEFAULT_DATA_PATH = "/mnt/02D0BBD8D0BBCFE1/repos/data/anyplace_synthetic_test_chunked"

def generate_cubic_pointcloud(
        block_size = 1.0,   # edge‐length of the cube along x, y, z
        center=(0.0, 0.0, 0.0),         # (cx, cy, cz) – cube's centre in world coordinates
        orientation=np.eye(3),          # 3 × 3 rotation matrix
        samples_per_edge=10,            # how many sample locations per axis
        surface_only=True              # if True, return only the outer faces (no interior points)
    ):
    """
    Build an (N, 3) NumPy array containing a point-cloud of a (possibly) 
    rectangular cube, then rotate and translate it.

    Parameters
    ----------
    side_lengths : 3-tuple of floats
        Physical size of the cube along (x, y, z) *before* rotation.
    center : 3-tuple of floats
        Coordinates of the cube's centre *after* rotation.
    orientation : (3, 3) ndarray
        A rotation matrix.  Use e.g. `scipy.spatial.transform.Rotation`
        or any other method you like to create it.
    samples_per_edge : int
        Number of samples per axis.  When ``surface_only`` is ``False`` the
        returned cloud will contain `samples_per_edge ** 3` points.
    surface_only : bool, optional
        If ``True`` return only the points that lie on the six faces of the
        cube (including edges and corners).  Default is ``True``.

    Returns
    -------
    pc : (N, 3) ndarray
        The point-cloud in world coordinates (after rotation & translation).
    """
    lx, ly, lz = block_size, block_size, block_size

    # Build a regular grid centred at the origin.
    x = np.linspace(-lx/2, lx/2, samples_per_edge)
    y = np.linspace(-ly/2, ly/2, samples_per_edge)
    z = np.linspace(-lz/2, lz/2, samples_per_edge)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    if surface_only:
        # Points that lie on any of the six faces: at least one coordinate
        # is on the corresponding extreme value.
        face_mask = (
            np.isclose(xx, x[0]) | np.isclose(xx, x[-1]) |
            np.isclose(yy, y[0]) | np.isclose(yy, y[-1]) |
            np.isclose(zz, z[0]) | np.isclose(zz, z[-1])
        )

        cube_coords = np.stack((xx[face_mask], yy[face_mask], zz[face_mask]), axis=1)
    else:
        # Stack to a flat (N, 3) array (full grid).
        cube_coords = np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=1)

    # Rotate and translate.
    world_coords = cube_coords @ orientation.T + np.asarray(center)

    return world_coords




def generate_random_block_positions(
    base_y=0.0, block_size=1.0
):
    """
    Generate random positions for two blocks with one block space between them.

    Args:
        base_y: Base Y coordinate for the blocks
        block_size: Size of each block (auto-detected from URDF if None)
        urdf_path: Path to the URDF file to measure
        scaling: Scaling factor applied to the URDF

    Returns:
        Tuple of (left_block_pos, right_block_pos, target_pos)
    """
    # Auto-detect block size from URDF if not provided
    # if block_size is None:
    #     block_size = get_urdf_bounding_box(urdf_path, scaling)
    #     print(f"Detected block size from URDF: {block_size:.4f} meters")
        # Generate random position and orientation for the left block
    left_x = random.uniform(0.2, 0.35)
    left_pos = [left_x, base_y, 0]

    # Generate same random orientation for both stationary blocks
    block_orientation = [
        random.uniform(-0.3, 0.3),  # Random roll
        random.uniform(-0.3, 0.3),  # Random pitch
        random.uniform(-np.pi, np.pi),  # Random yaw
    ]

    # Calculate right block position based on left block's X-axis direction
    # Convert Euler angles to rotation matrix to get X-axis direction

    rotation_matrix = R.from_euler("xyz", block_orientation).as_matrix()
    x_axis_direction = rotation_matrix[:, 0]  # First column is X-axis direction

    # Position right block one block size away along left block's X-axis
    displacement = x_axis_direction * block_size * 2
    right_pos = [
        left_pos[0] + displacement[0],
        left_pos[1] + displacement[1],
        left_pos[2] + displacement[2],
    ]

    # Target position is halfway between left and right blocks
    target_pos = [
        (left_pos[0] + right_pos[0]) / 2,
        (left_pos[1] + right_pos[1]) / 2,
        (left_pos[2] + right_pos[2]) / 2,
    ]

    return left_pos, right_pos, target_pos, block_orientation



def save_robotics_data_npz(
    filename,
    success,
    multi_obj_names_parent,
    multi_obj_names_child,
    multi_obj_start_pcd_parent,
    multi_obj_start_pcd_child,
    multi_obj_final_pcd_parent,
    multi_obj_final_pcd_child,
    cam_intrinsics,
    cam_poses,
    real_sim,
    multi_obj_start_obj_pose_parent,
    multi_obj_start_obj_pose_child,
    multi_obj_final_obj_pose_parent,
    multi_obj_final_obj_pose_child,
    # Optional parameters (can be None)
    grasp_pose_world_parent=None,
    grasp_pose_world_child=None,
    place_pose_world_parent=None,
    place_pose_world_child=None,
    grasp_joints_parent=None,
    grasp_joints_child=None,
    place_joints_parent=None,
    place_joints_child=None,
    ee_link=None,
    gripper_type=None,
    pcd_pts=None,
    processed_pcd=None,
    rgb_imgs=None,
    depth_imgs=None,
    multi_obj_part_pose_dict=None,
    compressed=True
):
    """
    Save robotics data to NPZ file with proper handling of None values.
    
    Required Parameters:
        filename (str): Output NPZ filename
        success (bool): Success flag
        multi_obj_names_parent (str): Parent object name
        multi_obj_names_child (str): Child object name
        multi_obj_start_pcd_parent (np.array): Parent start point cloud (N, 3)
        multi_obj_start_pcd_child (np.array): Child start point cloud (M, 3)
        multi_obj_final_pcd_parent (np.array): Parent final point cloud (N, 3)
        multi_obj_final_pcd_child (np.array): Child final point cloud (M, 3)
        cam_intrinsics (np.array): Camera intrinsics (4, 3, 3)
        cam_poses (np.array): Camera poses (4, 4, 4)
        real_sim (str): 'real' or 'sim'
        multi_obj_start_obj_pose_parent (np.array): Parent start pose (7,)
        multi_obj_start_obj_pose_child (np.array): Child start pose (7,)
        multi_obj_final_obj_pose_parent (np.array): Parent final pose (7,)
        multi_obj_final_obj_pose_child (np.array): Child final pose (7,)
    
    Optional Parameters (default=None):
        All other parameters that can be None
        
    Returns:
        None (saves file to disk)
    """
    
    def _convert_value(value):
        """Convert value to numpy array format suitable for NPZ"""
        if value is None:
            return np.array(['None'], dtype='U10')
        elif isinstance(value, bool):
            return np.array(value)
        elif isinstance(value, str):
            return np.array(value, dtype='U50')
        elif isinstance(value, (int, float)):
            return np.array(value)
        elif isinstance(value, np.ndarray):
            return value.astype(np.float32) if value.dtype.kind == 'f' else value
        else:
            return np.array(value)
    
    # Build the data dictionary
    data_dict = {
        "0": {
            'success': _convert_value(success),
            'multi_obj_names': np.array({
                'parent': _convert_value(multi_obj_names_parent),
                'child': _convert_value(multi_obj_names_child),
            }),
            'multi_obj_start_pcd' : np.array({
                'parent': _convert_value(multi_obj_start_pcd_parent),
                'child': _convert_value(multi_obj_start_pcd_child),
            }),
            'multi_obj_final_pcd' : np.array({
                'parent': _convert_value(multi_obj_final_pcd_parent),
                'child': _convert_value(multi_obj_final_pcd_child),
            }),
            'cam_intrinsics': _convert_value(cam_intrinsics),
            'cam_poses': _convert_value(cam_poses),
            'real_sim': _convert_value(real_sim),
            'multi_obj_start_obj_pose' : np.array({
                'parent': _convert_value(multi_obj_start_obj_pose_parent),
                'child': _convert_value(multi_obj_start_obj_pose_child),
            }),
            'multi_obj_final_obj_pose' : np.array({
                'parent': _convert_value(multi_obj_final_obj_pose_parent),
                'child': _convert_value(multi_obj_final_obj_pose_child),
            }),
            
            # # Optional fields
            'grasp_pose_world' : np.array({
                'parent': _convert_value(grasp_pose_world_parent),
                'child': _convert_value(grasp_pose_world_child),
            }),
            'place_pose_world' : np.array({
                'parent': _convert_value(place_pose_world_parent),
                'child': _convert_value(place_pose_world_child),
            }),
            'grasp_joints' : np.array({
                'parent': _convert_value(grasp_joints_parent),
                'child': _convert_value(grasp_joints_child),
            }),
            'place_joints' : np.array({
                'parent': _convert_value(place_joints_parent),
                'child': _convert_value(place_joints_child),
            }),

            'ee_link': _convert_value(ee_link),
            'gripper_type': _convert_value(gripper_type),
            'pcd_pts': _convert_value(pcd_pts),
            'processed_pcd': _convert_value(processed_pcd),
            'rgb_imgs': _convert_value(rgb_imgs),
            'depth_imgs': _convert_value(depth_imgs),
            'multi_obj_part_pose_dict': _convert_value(multi_obj_part_pose_dict),
        }
        # Required fields
    }
    
    # Save to NPZ file
    if compressed:
        np.savez_compressed(filename, **data_dict)
    else:
        np.savez(filename, **data_dict)
    
    print(f"Robotics data saved to {filename}")





if __name__ == "__main__":
    num_samples = 2000   
    train_split = 1800
    val_split = 100
    test_split = 100
    block_size = 0.05
    n_samples_per_edge = 30
    sim = setup_placement_scenario()

    intrinsics = np.array([sim.realsensed435_cam[i]["intrinsics"] for i in range(4)])
    cam_poses = []
    for i in range(4):
        cam_pose = np.eye(4)
        cam_pose[:3,:3] = R.from_quat(sim.realsensed435_cam[i]["rotation"]).as_matrix()
        cam_pose[:3,3] = sim.realsensed435_cam[i]["position"]
        cam_poses.append(cam_pose)


    if not os.path.exists(f"{DEFAULT_DATA_PATH}/0"):
        os.makedirs(f"{DEFAULT_DATA_PATH}/0", exist_ok=True)
    if not os.path.exists(f"{DEFAULT_DATA_PATH}/split_info"):
        os.makedirs(f"{DEFAULT_DATA_PATH}/split_info", exist_ok=True)


    # Generate filenames for each split
    train_filenames = [f"demo_aug_{i}.npz" for i in range(train_split)]
    val_filenames = [f"demo_aug_{i}.npz" for i in range(train_split, train_split + val_split)]
    test_filenames = [f"demo_aug_{i}.npz" for i in range(train_split + val_split, num_samples)]

    # Write train split
    with open(os.path.join(f"{DEFAULT_DATA_PATH}/split_info/train_split.txt"), "w") as f:
        f.write("\n".join(train_filenames))

    # Write validation split 
    with open(os.path.join(f"{DEFAULT_DATA_PATH}/split_info/train_val_split.txt"), "w") as f:
        f.write("\n".join(val_filenames))

    # Write test split
    with open(os.path.join(f"{DEFAULT_DATA_PATH}/split_info/test_split.txt"), "w") as f:
        f.write("\n".join(test_filenames))
        


    for i in range(num_samples):
        left_pos, right_pos, target_pos, block_orientation = generate_random_block_positions(block_size=block_size)
        pcd_npy_left = generate_cubic_pointcloud(block_size=block_size,
                                            center=left_pos,
                                            orientation=R.from_euler("xyz", block_orientation).as_matrix(),
                                            samples_per_edge=n_samples_per_edge,
                                            surface_only=True)
        
        pcd_npy_right = generate_cubic_pointcloud(block_size=block_size,
                                            center=right_pos,
                                            orientation=R.from_euler("xyz", block_orientation).as_matrix(),
                                            samples_per_edge=n_samples_per_edge,
                                            surface_only=True)
        
        pcd_npy_target = generate_cubic_pointcloud(block_size=block_size,
                                            center=target_pos,
                                            orientation=R.from_euler("xyz", block_orientation).as_matrix(),
                                            samples_per_edge=n_samples_per_edge,
                                            surface_only=True)

        # ----------------------------- Random fourth block -----------------------------
        rand_pos = [random.uniform(-block_size*5, block_size*5) for _ in range(3)]
        rand_orientation_euler = [random.uniform(-np.pi, np.pi) for _ in range(3)]
        pcd_npy_rand = generate_cubic_pointcloud(block_size=block_size,
                                            center=rand_pos,
                                            orientation=R.from_euler("xyz", rand_orientation_euler).as_matrix(),
                                            samples_per_edge=n_samples_per_edge,
                                            surface_only=True)

        pcd_start_parent = o3d.geometry.PointCloud()
        pcd_start_parent.points = o3d.utility.Vector3dVector(pcd_npy_left)
        pcd_start_parent.points.extend(o3d.utility.Vector3dVector(pcd_npy_right))
        pcd_start_child = o3d.geometry.PointCloud()
        pcd_start_child.points = o3d.utility.Vector3dVector(pcd_npy_rand)


        pcd_end_parent = o3d.geometry.PointCloud()
        pcd_end_parent.points = o3d.utility.Vector3dVector(pcd_npy_left)
        pcd_end_parent.points.extend(o3d.utility.Vector3dVector(pcd_npy_right))
        pcd_end_child = o3d.geometry.PointCloud()
        pcd_end_child.points = o3d.utility.Vector3dVector(pcd_npy_target)

        # [x , y , z , qx , qy , qz , qw] -> multi_obj_start_obj_pose_parent

        parent_start_pos = target_pos
        parent_start_quat = R.from_euler("xyz", block_orientation).as_quat().tolist()
        child_start_pos = rand_pos
        child_start_quat = R.from_euler("xyz", rand_orientation_euler).as_quat().tolist()

        parent_end_pos = target_pos
        parent_end_quat = R.from_euler("xyz", block_orientation).as_quat().tolist()
        child_end_pos = target_pos
        child_end_quat = R.from_euler("xyz", block_orientation).as_quat().tolist()
        save_path = f"{DEFAULT_DATA_PATH}/0/demo_aug_{i}.npz"

        save_robotics_data_npz(
            filename=save_path,
            success=True,
            multi_obj_names_parent="parent",
            multi_obj_names_child="child",
            multi_obj_start_pcd_parent=pcd_start_parent.points,
            multi_obj_start_pcd_child=pcd_start_child.points,
            multi_obj_final_pcd_parent=pcd_end_parent.points,
            multi_obj_final_pcd_child=pcd_end_child.points,
            cam_intrinsics=intrinsics,
            cam_poses=cam_poses,
            real_sim="sim",
            multi_obj_start_obj_pose_parent=parent_start_pos + parent_start_quat,
            multi_obj_start_obj_pose_child=child_start_pos + child_start_quat,
            multi_obj_final_obj_pose_parent=parent_end_pos + parent_end_quat,
            multi_obj_final_obj_pose_child=child_end_pos + child_end_quat,
        )



        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pcd_npy_left)
        # pcd.points.extend(o3d.utility.Vector3dVector(pcd_npy_right))
        # pcd.points.extend(o3d.utility.Vector3dVector(pcd_npy_target))
        # pcd.points.extend(o3d.utility.Vector3dVector(pcd_npy_rand))
        # pb.addUserDebugPoints(
        #     np.asarray(pcd.points).tolist(),                         # point positions
        #     [[1, 0, 0]] * len(pcd.points),            # one RGB triplet per point
        #     pointSize=1,
        #     lifeTime=10000,
        # )
        # for i in range(4):
        #     color, depth, _ = sim.render_camera(camera_idx=i)
        #     # Display color image using matplotlib
        #     plt.figure(figsize=(10, 5))
        #     plt.subplot(121)
        #     plt.imshow(color)
        #     plt.title('Color Image')
        #     plt.axis('off')
        #     plt.show()
        # breakpoint()
        # o3d.visualization.draw_geometries([pcd])






