#!/usr/bin/env python3
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Compute forward kinematics (FK) for a LeRobot dataset recorded with the Trossen AI
stationary robot. Outputs end-effector poses for both arms (left/right gripper center)
and the head camera pose (center-top camera, not wrist cameras).

Uses the `pinocchio` package for FK (https://stack-of-tasks.github.io/pinocchio/).
Install with: pip install pin

The URDF is expected to be the stationary_ai.urdf from trossen_arm_description.
Joint states in the dataset are assumed to be in the order produced by the
Trossen AI stationary robot: left_joint_0..left_joint_6, right_joint_0..right_joint_6.
The first 6 joints per arm are used for arm FK (joint_6 is gripper and does not
affect the arm end-effector pose in the URDF used here).
"""

import argparse
import copy
import tempfile
from pathlib import Path
import os
import shutil

import numpy as np
import torch
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import pinocchio as pin
except ImportError:
    raise ImportError(
        "This script requires pinocchio for forward kinematics. Install with: pip install pin"
    )
from scipy.spatial.transform import Rotation

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.utils.io_utils import write_video


# Link (frame) names in the stationary_ai URDF
LEFT_EE_LINK = "follower_left_ee_gripper_link"
RIGHT_EE_LINK = "follower_right_ee_gripper_link"
HEAD_CAMERA_LINK = "cam_high_color_optical_frame"  # Center-top head camera optical frame (for RGB projection)

# URDF revolute joint names per arm (arm joints only; gripper is joint_6 and not in URDF arm chain for EE pose)
LEFT_URDF_JOINTS = [f"follower_left_joint_{i}" for i in range(6)]
RIGHT_URDF_JOINTS = [f"follower_right_joint_{i}" for i in range(6)]


def _load_urdf(urdf_path: str | Path, package_root: str | Path | None = None):
    """Load URDF with Pinocchio, optionally resolving package:// paths."""
    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    with open(urdf_path) as f:
        urdf_str = f.read()

    if package_root is not None:
        package_root = Path(package_root).resolve()
        urdf_str = urdf_str.replace(
            "package://trossen_arm_description", str(package_root)
        )

    if package_root is not None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".urdf", delete=False
        ) as tmp:
            tmp.write(urdf_str)
            tmp_path = tmp.name
        try:
            model = pin.buildModelFromUrdf(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        model = pin.buildModelFromUrdf(str(urdf_path))

    return model


def _build_state_to_q_mapping(model: pin.Model, state_names: list[str]) -> list[tuple[int, int]]:
    """
    Build mapping (state_idx, q_idx) so we can fill q from state.
    Returns list of (index_into_state_vector, index_into_q_vector) for each of the 12 arm joints.
    """
    name_to_state_idx = {n: i for i, n in enumerate(state_names)}
    mapping = []
    for j in range(6):
        left_key = f"left_joint_{j}"
        urdf_name = f"follower_left_joint_{j}"
        if left_key in name_to_state_idx and model.existJointName(urdf_name):
            joint_id = model.getJointId(urdf_name)
            idx_q = model.joints[joint_id].idx_q
            mapping.append((name_to_state_idx[left_key], idx_q))
    for j in range(6):
        right_key = f"right_joint_{j}"
        urdf_name = f"follower_right_joint_{j}"
        if right_key in name_to_state_idx and model.existJointName(urdf_name):
            joint_id = model.getJointId(urdf_name)
            idx_q = model.joints[joint_id].idx_q
            mapping.append((name_to_state_idx[right_key], idx_q))
    return mapping


def _pose_from_se3(placement: pin.SE3) -> tuple[np.ndarray, np.ndarray]:
    """Extract position (3,) and quaternion xyzw (4,) from Pinocchio SE3 placement."""
    position = np.array(placement.translation).reshape(3)
    R = placement.rotation
    quat_xyzw = Rotation.from_matrix(R).as_quat()
    return position, quat_xyzw


def compute_fk_for_frame(
    model: pin.Model,
    data: pin.Data,
    state: np.ndarray,
    state_names: list[str],
    state_to_q: list[tuple[int, int]],
    frame_ids: dict[str, int],
) -> dict:
    """
    Compute FK for one frame: left EE, right EE, and head camera pose.
    Returns dict with keys: left_ee_position, left_ee_quat_xyzw, right_ee_position,
    right_ee_quat_xyzw, head_camera_position, head_camera_quat_xyzw.
    """
    q = np.zeros(model.nq)
    for state_idx, q_idx in state_to_q:
        q[q_idx] = float(state[state_idx])

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    out = {}
    for name, key_pos, key_quat in [
        (LEFT_EE_LINK, "left_ee_position", "left_ee_quat_xyzw"),
        (RIGHT_EE_LINK, "right_ee_position", "right_ee_quat_xyzw"),
        (HEAD_CAMERA_LINK, "head_camera_position", "head_camera_quat_xyzw"),
    ]:
        fid = frame_ids[name]
        pos, quat = _pose_from_se3(data.oMf[fid])
        out[key_pos] = pos
        out[key_quat] = quat

    return out


def compute_fk_for_dataset(
    dataset: LeRobotDataset,
    urdf_path: str | Path,
    package_root: str | Path | None = None,
    output_path: str | Path | None = None,
):
    """
    Run FK for every frame in the dataset and optionally save results.
    Returns list of dicts (one per frame) with pose arrays.
    """
    model = _load_urdf(urdf_path, package_root=package_root)
    data = model.createData()

    state_key = "observation.state"
    if state_key not in dataset.features:
        raise KeyError(
            f"Dataset has no feature '{state_key}'. "
            "Ensure the dataset contains joint state observations."
        )
    state_names = dataset.features[state_key].get("names")
    if state_names is None:
        raise ValueError(
            f"Feature '{state_key}' has no 'names'. "
            "Cannot map state vector to URDF joint names."
        )

    state_to_q = _build_state_to_q_mapping(model, state_names)
    if len(state_to_q) != 12:
        raise ValueError(
            f"Expected 12 arm joints (6 left + 6 right) in the model; got {len(state_to_q)}. "
            "Check that the URDF and dataset state names match (left_joint_0..5, right_joint_0..5)."
        )
    frame_names = [LEFT_EE_LINK, RIGHT_EE_LINK, HEAD_CAMERA_LINK]
    frame_ids = {}
    for name in frame_names:
        try:
            frame_ids[name] = model.getFrameId(name)
        except Exception as e:
            raise ValueError(
                f"URDF has no frame named '{name}'. "
                "Check that the URDF is stationary_ai with links "
                "follower_left_ee_gripper_link, follower_right_ee_gripper_link, cam_high_color_optical_frame."
            ) from e

    results = []

    
    for idx in tqdm(range(len(dataset)), desc="FK", unit="frame"):
        frame = dataset[idx]
        state = frame[state_key]
        if hasattr(state, "numpy"):
            state = state.numpy()
        state = np.asarray(state, dtype=np.float64)
        if state.ndim > 1:
            state = state.ravel()
        pose_dict = compute_fk_for_frame(
            model, data, state, state_names, state_to_q, frame_ids
        )
        results.append(pose_dict)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            left_ee_position=np.array([r["left_ee_position"] for r in results]),
            left_ee_quat_xyzw=np.array([r["left_ee_quat_xyzw"] for r in results]),
            right_ee_position=np.array([r["right_ee_position"] for r in results]),
            right_ee_quat_xyzw=np.array([r["right_ee_quat_xyzw"] for r in results]),
            head_camera_position=np.array([r["head_camera_position"] for r in results]),
            head_camera_quat_xyzw=np.array([r["head_camera_quat_xyzw"] for r in results]),
        )
        print(f"Saved FK poses to {output_path}")

    return results


def _print_fk_results(results: list[dict], limit: int | None = 50) -> None:
    """Print FK poses for each frame: position (x,y,z) and quaternion (xyzw) for left EE, right EE, head camera."""
    to_show = results[:limit] if limit is not None else results
    print("\nframe_index | left_ee_position (x,y,z) | right_ee_position (x,y,z) | head_camera_position (x,y,z)")
    print("            | left_ee_quat_xyzw        | right_ee_quat_xyzw        | head_camera_quat_xyzw")
    print("-" * 120)
    for idx, r in enumerate(to_show):
        lp = r["left_ee_position"]
        rp = r["right_ee_position"]
        hp = r["head_camera_position"]
        lq = r["left_ee_quat_xyzw"]
        rq = r["right_ee_quat_xyzw"]
        hq = r["head_camera_quat_xyzw"]
        print(
            f"{idx:11} | [{lp[0]:.4f},{lp[1]:.4f},{lp[2]:.4f}] "
            f"| [{rp[0]:.4f},{rp[1]:.4f},{rp[2]:.4f}] "
            f"| [{hp[0]:.4f},{hp[1]:.4f},{hp[2]:.4f}]"
        )
        print(
            f"            | [{lq[0]:.4f},{lq[1]:.4f},{lq[2]:.4f},{lq[3]:.4f}] "
            f"| [{rq[0]:.4f},{rq[1]:.4f},{rq[2]:.4f},{rq[3]:.4f}] "
            f"| [{hq[0]:.4f},{hq[1]:.4f},{hq[2]:.4f},{hq[3]:.4f}]"
        )
    if limit is not None and len(results) > limit:
        print(f"... ({len(results) - limit} more frames)")
    print("-" * 120)
    print(f"Total: {len(results)} frames")


# Default camera intrinsics for head camera (640x480). Not in dataset metadata; adjust if you have calibration.
# Typical pinhole: cx, cy at image center; fx, fy ~ 600 for ~60 deg horizontal FOV at 640 width.
DEFAULT_FX = 381.092
DEFAULT_FY = 381.092
DEFAULT_CX = 310.085
DEFAULT_CY = 245.318


# Rectified 2 (640×480)

# fx = 381.092

# fy = 381.092

# cx = 310.085

# cy = 245.318


def _project_ee_to_image(
    cam_position: np.ndarray,
    cam_quat_xyzw: np.ndarray,
    left_ee_world: np.ndarray,
    right_ee_world: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """
    Project left and right end-effector 3D positions (world frame) into head camera image.
    Camera pose is in world frame (position + quat xyzw). Pinhole model: Z forward, X right, Y down.
    Returns (left_uv, right_uv) in pixel coords; None if point is behind camera (Z_cam <= 0).
    """
    R_world_cam = Rotation.from_quat(cam_quat_xyzw).as_matrix()
    t_cam = np.asarray(cam_position).reshape(3)

    def project(p_world: np.ndarray) -> tuple[int, int] | None:
        p_world = np.asarray(p_world).reshape(3)
        p_cam = R_world_cam.T @ (p_world - t_cam)
        if p_cam[2] <= 1e-6:
            return None
        u = fx * p_cam[0] / p_cam[2] + cx
        v = fy * p_cam[1] / p_cam[2] + cy
        return (int(round(u)), int(round(v)))

    left_uv = project(left_ee_world)
    right_uv = project(right_ee_world)
    return left_uv, right_uv


def _frame_tensor_to_numpy_rgb(frame: torch.Tensor) -> np.ndarray:
    """Convert dataset frame (C, H, W) tensor to (H, W, 3) uint8 numpy for drawing."""
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.dtype == np.float32 or frame.dtype == np.float64:
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def render_episode_video_with_ee(
    dataset: LeRobotDataset,
    episode_index: int,
    results: list[dict],
    camera_key: str,
    output_path: str | Path,
    fps: int,
    fx: float = DEFAULT_FX,
    fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX,
    cy: float = DEFAULT_CY,
) -> None:
    """
    Load head camera video for one episode, project both EEs onto each frame, draw them, and save video.
    """
    if camera_key not in dataset.meta.camera_keys:
        raise ValueError(
            f"Camera key '{camera_key}' not in dataset cameras: {dataset.meta.camera_keys}"
        )

    ep_from = dataset.episode_data_index["from"][episode_index].item()
    ep_to = dataset.episode_data_index["to"][episode_index].item()
    n_frames = ep_to - ep_from

    frames_out = []
    for i in tqdm(range(ep_from, ep_to), desc="Rendering video", unit="frame"):
        item = dataset[i]
        frame = item[camera_key]
        img = _frame_tensor_to_numpy_rgb(frame)

        r = results[i]
        left_uv, right_uv = _project_ee_to_image(
            r["head_camera_position"],
            r["head_camera_quat_xyzw"],
            r["left_ee_position"],
            r["right_ee_position"],
            fx, fy, cx, cy,
        )

        # Draw left EE in blue, right EE in red
        if left_uv is not None:
            cv2.circle(img, left_uv, 12, (255, 0, 0), 2)  # BGR blue
            cv2.putText(img, "L", (left_uv[0] + 14, left_uv[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        if right_uv is not None:
            cv2.circle(img, right_uv, 12, (0, 0, 255), 2)  # BGR red
            cv2.putText(img, "R", (right_uv[0] + 14, right_uv[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        frames_out.append(img)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(output_path), frames_out, fps=fps)
    print(f"Saved EE overlay video to {output_path} ({n_frames} frames, {fps} fps)")


def plot_3d_trajectories(
    results: list[dict],
    output_path: str | Path | None = None,
    show_head_camera: bool = False,
) -> None:
    """
    Plot 3D trajectories of left EE, right EE, and optionally head camera.
    Saves to file if output_path is provided, otherwise shows interactively.
    """
    left_positions = np.array([r["left_ee_position"] for r in results])
    right_positions = np.array([r["right_ee_position"] for r in results])
    head_positions = np.array([r["head_camera_position"] for r in results]) if show_head_camera else None

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot trajectories
    ax.plot(left_positions[:, 0], left_positions[:, 1], left_positions[:, 2], 
            "b-", label="Left EE", linewidth=1.5, alpha=0.7)
    ax.plot(right_positions[:, 0], right_positions[:, 1], right_positions[:, 2], 
            "r-", label="Right EE", linewidth=1.5, alpha=0.7)
    
    if head_positions is not None:
        ax.plot(head_positions[:, 0], head_positions[:, 1], head_positions[:, 2], 
                "g-", label="Head Camera", linewidth=1.0, alpha=0.5)

    # Mark start and end points
    ax.scatter(left_positions[0, 0], left_positions[0, 1], left_positions[0, 2], 
              c="blue", marker="o", s=100, label="Left EE Start", edgecolors="black", linewidths=1)
    ax.scatter(left_positions[-1, 0], left_positions[-1, 1], left_positions[-1, 2], 
              c="blue", marker="s", s=100, label="Left EE End", edgecolors="black", linewidths=1)
    ax.scatter(right_positions[0, 0], right_positions[0, 1], right_positions[0, 2], 
              c="red", marker="o", s=100, label="Right EE Start", edgecolors="black", linewidths=1)
    ax.scatter(right_positions[-1, 0], right_positions[-1, 1], right_positions[-1, 2], 
              c="red", marker="s", s=100, label="Right EE End", edgecolors="black", linewidths=1)

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_zlabel("Z (m)", fontsize=12)
    ax.set_title("3D End-Effector Trajectories", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set equal aspect ratio
    all_positions = np.vstack([left_positions, right_positions])
    if head_positions is not None:
        all_positions = np.vstack([all_positions, head_positions])
    max_range = np.array([
        all_positions[:, 0].max() - all_positions[:, 0].min(),
        all_positions[:, 1].max() - all_positions[:, 1].min(),
        all_positions[:, 2].max() - all_positions[:, 2].min(),
    ]).max() / 2.0
    mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
    mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) * 0.5
    mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved 3D trajectory plot to {output_path}")
    else:
        plt.show()


def _default_urdf_path() -> Path:
    """Default URDF path: trossen_arm_description/urdf/generated/stationary_ai.urdf (sibling of lerobot)."""
    return Path(__file__).resolve().parent.parent.parent / "trossen_arm_description" / "urdf" / "generated" / "stationary_ai.urdf"


def _default_package_root() -> Path:
    """Default package root for resolving package://trossen_arm_description in the URDF."""
    return Path(__file__).resolve().parent.parent.parent / "trossen_arm_description"


def main():
    default_repo_id = "ykorkmaz/aloha_play_dataset_part_3"
    default_urdf = "/root/lerobot/data_processing/stationary_ai.urdf" #str(_default_urdf_path())
    default_package_root = str(_default_package_root())

    parser = argparse.ArgumentParser(
        description="Compute FK (end-effector and head camera poses) from a LeRobot dataset with joint states."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=default_repo_id,
        help=f"LeRobot dataset repo id (default: {default_repo_id}).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Path to local dataset root (if loading from disk instead of Hub).",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=default_urdf,
        help=f"Path to stationary_ai.urdf (default: trossen_arm_description/urdf/generated/stationary_ai.urdf next to lerobot).",
    )
    parser.add_argument(
        "--package-root",
        type=str,
        default=default_package_root,
        help="Path to trossen_arm_description package root for resolving package:// in URDF (default: sibling of lerobot).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="If set, save poses to this .npz file; otherwise only print to stdout.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of episode indices to process (default: all).",
    )
    parser.add_argument(
        "--print-limit",
        type=int,
        default=50,
        help="Max number of frames to print (default: 50). Use 0 to print all.",
    )
    parser.add_argument(
        "--render-video",
        action="store_true",
        default=True,
        help="Render one episode with EE points drawn on the head camera video.",
    )
    parser.add_argument(
        "--episode-for-video",
        type=int,
        default=0,
        help="Episode index to use for --render-video (default: 0).",
    )
    parser.add_argument(
        "--video-output",
        type=str,
        default="ee_overlay_ep0.mp4",
        help="Output path for the EE overlay video (default: ee_overlay_ep0.mp4).",
    )
    parser.add_argument(
        "--plot-3d",
        action="store_true",
        default=False,
        help="Plot 3D trajectories of end-effectors.",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="Output path for the 3D trajectory plot (default: show interactively).",
    )
    parser.add_argument(
        "--plot-head-camera",
        action="store_true",
        help="Include head camera trajectory in the 3D plot.",
    )
    args = parser.parse_args()

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        episodes=args.episodes if args.episodes else None,
        
    )


    print(f"Loaded dataset: {dataset.num_frames} frames, {dataset.num_episodes} episodes.")

    # import ipdb; ipdb.set_trace()
    results = compute_fk_for_dataset(
        dataset,
        urdf_path=args.urdf,
        package_root=args.package_root,
        output_path=args.output,
    )

    # Create a new dataset from scratch with add_frame / save_episode, then push to hub.
    output_repo_id = "ishika/aloha_play_dataset_part_3_with_fk_final"
    new_fk_features = {
        "left_ee_position": {"dtype": "float32", "shape": [3], "description": "FK left EE position [m]"},
        "left_ee_quat_xyzw": {"dtype": "float32", "shape": [4], "description": "FK left EE quat [x,y,z,w]"},
        "right_ee_position": {"dtype": "float32", "shape": [3], "description": "FK right EE position [m]"},
        "right_ee_quat_xyzw": {"dtype": "float32", "shape": [4], "description": "FK right EE quat [x,y,z,w]"},
        "head_camera_position": {"dtype": "float32", "shape": [3], "description": "FK head camera position [m]"},
        "head_camera_quat_xyzw": {"dtype": "float32", "shape": [4], "description": "FK head camera quat [x,y,z,w]"},
    }
    features_with_fk = copy.deepcopy(dataset.meta.info["features"])
    for k, v in new_fk_features.items():
        v = copy.deepcopy(v)
        v["shape"] = tuple(v["shape"])
        features_with_fk[k] = v
    # features_with_fk["timestamp"] = {"dtype": "float32", "shape": [], "description": "Timestamp [s]"}
    build_root = '/tmp/lerobot_fk_123'
    if os.path.exists(build_root):
        shutil.rmtree(build_root)
    print(f"Creating new dataset at {build_root} with add_frame/save_episode...")
    new_dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=dataset.meta.fps,
        root=build_root,
        robot_type=dataset.meta.robot_type,
        features=features_with_fk,
        use_videos=True,
    )

    ep_from = dataset.episode_data_index["from"]
    ep_to = dataset.episode_data_index["to"]
    for ep_idx in tqdm(range(dataset.num_episodes), desc="Episodes"):
        from_idx = int(ep_from[ep_idx].item())
        to_idx = int(ep_to[ep_idx].item())
        to_idx = min(to_idx, from_idx + 10)
        print(f"Adding episode {ep_idx} from {from_idx} to {to_idx}")
        print(f"Number of frames: {to_idx - from_idx}")
        for global_idx in tqdm(range(from_idx, to_idx), desc="Frames"):
            item = dataset[global_idx]
            fk = results[global_idx]
            frame = {
                "action": item["action"],
                "observation.state": item["observation.state"],
                "task": item["task"],
                "timestamp": np.float32(item["timestamp"].item()),
                "left_ee_position": fk["left_ee_position"].astype(np.float32),
                "left_ee_quat_xyzw": fk["left_ee_quat_xyzw"].astype(np.float32),
                "right_ee_position": fk["right_ee_position"].astype(np.float32),
                "right_ee_quat_xyzw": fk["right_ee_quat_xyzw"].astype(np.float32),
                "head_camera_position": fk["head_camera_position"].astype(np.float32),
                "head_camera_quat_xyzw": fk["head_camera_quat_xyzw"].astype(np.float32),
            }
            for cam_key in dataset.meta.camera_keys:
                frame[cam_key] = item[cam_key].permute(1, 2, 0)
            new_dataset.add_frame(frame)
        # new_dataset.episode_buffer["timestamp"] = new_dataset.episode_buffer["timestamp"].reshape(-1)
        new_dataset.save_episode()

    print("Pushing new dataset to the Hub...")
    new_dataset.push_to_hub(commit_message="Dataset with FK columns (add_frame/save_episode)")
    print(f"Pushed to https://huggingface.co/datasets/{output_repo_id}")

    import ipdb; ipdb.set_trace()
    dataset_with_fk = LeRobotDataset(output_repo_id, episodes=args.episodes if args.episodes else None)

    print(f"Computed FK for {len(results)} frames.")
    print_limit = None if args.print_limit == 0 else args.print_limit
    _print_fk_results(results, limit=print_limit)

    if args.render_video:
        ep_idx = args.episode_for_video
        if ep_idx < 0 or ep_idx >= dataset.num_episodes:
            print(f"Warning: --episode-for-video {ep_idx} is out of range [0, {dataset.num_episodes - 1}]. Skipping render.")
        else:
            camera_key = "observation.images.cam_high"
            render_episode_video_with_ee(
                dataset,
                episode_index=ep_idx,
                results=results,
                camera_key=camera_key,
                output_path=args.video_output,
                fps=dataset.meta.fps,
            )

    if args.plot_3d:
        plot_3d_trajectories(
            results,
            output_path=args.plot_output,
            show_head_camera=args.plot_head_camera,
        )


if __name__ == "__main__":
    main()
