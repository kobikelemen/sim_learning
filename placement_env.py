import numpy as np
import pybullet as pb
import pybullet_data
import random
from sim import (
    SimGrasp,
    ObjectInfo,
    CUBE_SCALING,
)
from scipy.spatial.transform import Rotation as R
import cv2
import matplotlib.pyplot as plt
import open3d as o3d

URDF_PATH = "/mnt/02D0BBD8D0BBCFE1/repos/gb_example/trossen_arm_description/urdf/generated/wxai/wxai_follower.urdf"

def get_urdf_bounding_box(urdf_path: str, scaling: float = 1.0) -> float:
    """
    Get the bounding box size of a URDF object.

    Args:
        urdf_path: Path to the URDF file
        scaling: Scaling factor applied to the object

    Returns:
        Approximate size (width) of the object
    """
    # Create temporary physics client to load the object
    temp_client = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())

    try:
        # Load the object
        obj_id = pb.loadURDF(urdf_path, globalScaling=scaling)

        # Get the AABB (Axis-Aligned Bounding Box)
        aabb_min, aabb_max = pb.getAABB(obj_id)

        # Calculate the size in X direction (width)
        size_x = aabb_max[0] - aabb_min[0]

        pb.disconnect(temp_client)
        return size_x

    except Exception as e:
        print(
            f"Warning: Could not load URDF {urdf_path}, using default size. Error: {e}"
        )
        pb.disconnect(temp_client)
        return 0.05  # Default fallback size


def generate_random_block_positions(
    base_y=0.0, block_size=None, urdf_path="cube_small.urdf", scaling=CUBE_SCALING
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
    if block_size is None:
        block_size = get_urdf_bounding_box(urdf_path, scaling)
        print(f"Detected block size from URDF: {block_size:.4f} meters")
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

def setup_placement_scenario():
        """
        Sets up a placement scenario with three blocks - two stationary blocks and one target block.
        
        Returns:
            SimGrasp: Configured simulation environment with the blocks placed
        """
        # Generate random positions for the blocks (auto-detect size from URDF)
        left_block_pos, right_block_pos, target_block_pos, block_orientation = (
            generate_random_block_positions(
                urdf_path="cube_small.urdf", scaling=CUBE_SCALING
            )
        )

        print(f"Left block position: {left_block_pos}")
        print(f"Right block position: {right_block_pos}")
        print(f"Target block position (for robot to place): {target_block_pos}")

        # Create different colored blocks for visual distinction
        left_block_color = [1, 0, 0, 1]  # Red
        right_block_color = [0, 0, 1, 1]  # Blue
        target_block_color = [0, 1, 0, 1]  # Green (the one robot should move)

        return SimGrasp(
            urdf_path=URDF_PATH,
            frequency=30,
            objects=[
                # Left LEGO block (same random orientation as right)
                ObjectInfo(
                    urdf_path="cube_small.urdf",
                    position=left_block_pos,
                    orientation=block_orientation,
                    scaling=CUBE_SCALING,
                    color=left_block_color,
                ),
                # Right LEGO block (same random orientation as left)
                ObjectInfo(
                    urdf_path="cube_small.urdf",
                    position=right_block_pos,
                    orientation=block_orientation,
                    scaling=CUBE_SCALING,
                    color=right_block_color,
                ),
                # Target block that robot should move to the middle (keep standard orientation for easier grasping)
                ObjectInfo(
                    urdf_path="cube_small.urdf",
                    position=[
                        target_block_pos[0],
                        target_block_pos[1] - 0.1,
                        target_block_pos[2],
                    ],  # Place it slightly away
                    orientation=[
                        random.uniform(-0.3, 0.3),  # Random roll
                        random.uniform(-0.3, 0.3),  # Random pitch
                        random.uniform(-np.pi, np.pi),  # Random yaw
                    ],
                    scaling=CUBE_SCALING,
                    color=target_block_color,
                ),
            ],
        )


def sample_mesh_points(body_uid, link_idx=-1, n_pts=50_000, random_seed=None):
    """Return ~n_pts points sampled from an object's visual mesh.

    Assumes `pb.getMeshData` returns the simplified two-tuple
    ``(num_vertices, vertices)``.
    """

    if random_seed is not None:
        np.random.seed(random_seed)
    breakpoint()
    # Get (num_vertices, vertex_list)
    num_vtx, verts_list = pb.getMeshData(
        body_uid, link_idx, flags=pb.MESH_DATA_SIMULATION_MESH
    )

    breakpoint()

    # Convert to (N,3) array
    vtx = np.asarray(verts_list, dtype=np.float32).reshape(-1, 3)
    if vtx.size == 0:
        return np.empty((0, 3))

    # Uniformly sample from existing vertices (with replacement if needed)
    if n_pts <= len(vtx):
        idxs = np.random.choice(len(vtx), size=n_pts, replace=False)
    else:
        idxs = np.random.choice(len(vtx), size=n_pts, replace=True)
    pts = vtx[idxs]

    return pts


# ----------------------------- Box sampling -----------------------------

def sample_box_points(body_uid, link_idx=-1, n_pts=50_000, random_seed=None):
    """Return n_pts points uniformly distributed *inside* a box primitive.

    Works for objects whose visual shape is a GEOM_BOX. Uses half-extents
    reported by `getVisualShapeData` to build the axis-aligned box in the
    link frame, then transforms points to world coordinates.
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    # Locate the visual shape entry corresponding to the link
    shape_entries = pb.getVisualShapeData(body_uid)
    half_extents = None
    for entry in shape_entries:
        _b, link, shape_type, dims = entry[:4]
        if link == link_idx and shape_type == pb.GEOM_BOX:
            half_extents = dims  # dims is (hx, hy, hz)
            break

    if half_extents is None:
        # Not a box or wrong link; return empty cloud
        return np.empty((0, 3))

    hx, hy, hz = half_extents

    # Sample in local box coordinates
    local_pts = np.random.uniform(
        low=[-hx, -hy, -hz], high=[hx, hy, hz], size=(n_pts, 3)
    )

    # Transform to world coordinates
    if link_idx == -1:
        pos, orn = pb.getBasePositionAndOrientation(body_uid)
    else:
        # Get link world pose
        ls = pb.getLinkState(body_uid, link_idx, computeForwardKinematics=True)
        pos, orn = ls[4], ls[5]

    # 3×3 rotation matrix from quaternion
    rot = np.asarray(pb.getMatrixFromQuaternion(orn)).reshape(3, 3)
    world_pts = (local_pts @ rot.T) + np.asarray(pos)

    return world_pts

def get_ground_truth_pointcloud(
    client,
    body_id: int,
    link_indices=None,
    flags=pb.MESH_DATA_SIMULATION_MESH,  # use p.MESH_DATA_VISUAL_MESH for the visual geometry
):
    """
    Return an (N, 3) NumPy array with *world-space* vertices of `body_id`.

    Parameters
    ----------
    client : the pybullet module or a BulletClient instance
    body_id : int
        ID returned by loadURDF/loadMJCF/etc.
    link_indices : iterable[int] or None
        Which links to include.  None = base (-1) + every joint link.
    flags : int
        MESH_DATA_SIMULATION_MESH (convex collision mesh) or
        MESH_DATA_VISUAL_MESH (high-res visual mesh).

    Notes
    -----
    • Works for multi-link models – every link is individually transformed
      into the world frame.  
    • Primitive shapes that do not expose a triangle mesh via
      `getMeshData` are silently skipped.  
    """
    if link_indices is None:
        link_indices = [-1] + list(range(client.getNumJoints(body_id)))

    pcs = []
    for link in link_indices:
        vcount, vertices = client.getMeshData(
            bodyUniqueId=body_id,
            linkIndex=link,
            flags=flags,
        )

        # Some primitives (box, sphere, cylinder) may return zero vertices
        if vcount == 0:
            continue

        v_local = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)

        # Pose of this link in the world frame
        if link == -1:
            pos, orn = client.getBasePositionAndOrientation(body_id)
        else:
            # indices 4 & 5 are FK-computed world pose of the link frame
            link_state = client.getLinkState(body_id, link, computeForwardKinematics=True)
            pos, orn = link_state[4], link_state[5]

        rot = np.array(client.getMatrixFromQuaternion(orn), dtype=np.float32).reshape(3, 3)
        v_world = (rot @ v_local.T).T + np.asarray(pos, dtype=np.float32)
        pcs.append(v_world)

    return np.concatenate(pcs, axis=0) if pcs else np.empty((0, 3), dtype=np.float32)



if __name__ == "__main__":
    sim = setup_placement_scenario()

    print("Scenario setup complete!")
    print("- Red block (left)")
    print("- Blue block (right)")
    print("- Green block (to be moved to the middle)")
    print(
        "Robot's task: Move the green block to fill the gap between red and blue blocks"
    )
    color, depth, _ = sim.render_camera()
    pcd = sim.create_pointcloud(color, depth)
    # Prepare visual prompt: segment objects, create masks, and mark them on the image.
    image, seg = sim.obs['image'], sim.obs['seg']
    obj_ids = np.unique(seg)[1:]
    
    breakpoint()
    pts = sample_box_points(3, -1)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(pts)
    o3d.visualization.draw_geometries([pcd_gt])

    # for obj_id in obj_ids:
    #     # Decode segmentation value into body/link.
    #     # body_uid = seg_val & ((1 << 24) - 1)
    #     # link_idx = (seg_val >> 24) - 1

    #     # Determine if this (body, link) is a primitive box
    #     shape_entries = pb.getVisualShapeData(obj_id)
    #     breakpoint()
    #     is_box = False
    #     for entry in shape_entries:
    #         _b, link, shape_type = entry[:3]
    #         if shape_type == pb.GEOM_BOX:
    #             pcd_gt = o3d.geometry.PointCloud()
    #             pcd_gt.points = o3d.utility.Vector3dVector(pts)
    #             o3d.visualization.draw_geometries([pcd_gt])
    #     # if is_box:
           
    #     # else:
    #     #     pts = sample_mesh_points(body_uid, link_idx)

        
    all_masks = np.stack([seg == objID for objID in obj_ids])
    # Keep simulation running
    while True:
        sim.step_simulation()
