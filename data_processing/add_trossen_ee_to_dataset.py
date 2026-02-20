#!/usr/bin/env python3
"""
Add end-effector positions (world/robot root frame) to a Trossen LeRobot dataset
using forward kinematics. One point per arm (left gripper, right gripper).

Output is saved to an HDF5 file that can be used alongside the original dataset.
The HDF5 contains:
  - left_ee_position: (N, 3) in world frame
  - right_ee_position: (N, 3) in world frame
  - head_camera_position: (N, 3) for transforming EE to camera frame
  - head_camera_quat_xyzw: (N, 4)
  - episode_from, episode_to: (num_episodes,) for indexing

Requires: pinocchio, lerobot, scipy. Install with:
  pip install pin lerobot scipy
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import h5py
from tqdm import tqdm

try:
    import pinocchio as pin
except ImportError:
    raise ImportError("This script requires pinocchio. Install with: pip install pin")
from scipy.spatial.transform import Rotation

# LeRobot dataset (optional import so script can document usage without lerobot)
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    LeRobotDataset = None

# Link names in the stationary_ai URDF (same as compute_fk_from_dataset.py)
LEFT_EE_LINK = "follower_left_ee_gripper_link"
RIGHT_EE_LINK = "follower_right_ee_gripper_link"
HEAD_CAMERA_LINK = "cam_high_color_optical_frame"

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
    """Build mapping (state_idx, q_idx) for the 12 arm joints."""
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
    Compute FK for one frame: left EE, right EE, and head camera pose (world frame).
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


def run_fk_and_save(
    dataset: "LeRobotDataset",
    urdf_path: str | Path,
    package_root: str | Path | None = None,
    output_path: str | Path = None,
) -> Path:
    """
    Run FK for every frame and save to HDF5.
    Saves: left_ee_position, right_ee_position, head_camera_position, head_camera_quat_xyzw,
    and episode_from, episode_to for indexing.
    """
    if LeRobotDataset is None:
        raise ImportError("lerobot is required. Install with: pip install lerobot")

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
            f"Expected 12 arm joints (6 left + 6 right); got {len(state_to_q)}. "
            "Check that the URDF and dataset state names match."
        )

    frame_names = [LEFT_EE_LINK, RIGHT_EE_LINK, HEAD_CAMERA_LINK]
    frame_ids = {}
    for name in frame_names:
        try:
            frame_ids[name] = model.getFrameId(name)
        except Exception as e:
            raise ValueError(
                f"URDF has no frame named '{name}'."
            ) from e

    num_frames = len(dataset)
    left_ee = np.zeros((num_frames, 3), dtype=np.float64)
    right_ee = np.zeros((num_frames, 3), dtype=np.float64)
    head_pos = np.zeros((num_frames, 3), dtype=np.float64)
    head_quat = np.zeros((num_frames, 4), dtype=np.float64)

    for idx in tqdm(range(num_frames), desc="FK", unit="frame"):
        frame = dataset[idx]
        state = frame[state_key]
        if hasattr(state, "numpy"):
            state = state.numpy()
        state = np.asarray(state, dtype=np.float64)
        if state.ndim > 1:
            state = state.ravel()

        pose = compute_fk_for_frame(
            model, data, state, state_names, state_to_q, frame_ids
        )
        left_ee[idx] = pose["left_ee_position"]
        right_ee[idx] = pose["right_ee_position"]
        head_pos[idx] = pose["head_camera_position"]
        head_quat[idx] = pose["head_camera_quat_xyzw"]

    # Episode boundaries for dataloader indexing
    ep_from = dataset.episode_data_index["from"].numpy()
    ep_to = dataset.episode_data_index["to"].numpy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("left_ee_position", data=left_ee)
        f.create_dataset("right_ee_position", data=right_ee)
        f.create_dataset("head_camera_position", data=head_pos)
        f.create_dataset("head_camera_quat_xyzw", data=head_quat)
        f.create_dataset("episode_from", data=ep_from)
        f.create_dataset("episode_to", data=ep_to)
        f.attrs["num_frames"] = num_frames
        f.attrs["num_episodes"] = len(ep_from)

    print(f"Saved EE and camera poses to {output_path} ({num_frames} frames, {len(ep_from)} episodes)")
    return output_path


# Same defaults as lerobot/script/compute_fk_from_dataset.py
DEFAULT_REPO_ID = "ykorkmaz/aloha_play_dataset_part_3"


def _default_urdf_path() -> Path:
    """Default URDF path: trossen_arm_description/urdf/generated/stationary_ai.urdf (sibling of this repo)."""
    # return Path(__file__).resolve().parent.parent.parent / "trossen_arm_description" / "urdf" / "generated" / "stationary_ai.urdf"
    return "/root/lerobot/data_processing/stationary_ai.urdf"


def _default_package_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "trossen_arm_description"


def main():
    parser = argparse.ArgumentParser(
        description="Add end-effector positions (FK) to a Trossen LeRobot dataset. Saves to HDF5."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"LeRobot dataset repo id (default: {DEFAULT_REPO_ID}, same as compute_fk_from_dataset.py).",
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
        default=None,
        help="Path to stationary_ai.urdf. Default: trossen_arm_description/urdf/generated/stationary_ai.urdf next to this repo.",
    )
    parser.add_argument(
        "--package-root",
        type=str,
        default=None,
        help="Path to trossen_arm_description package root for package:// in URDF.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HDF5 path. Default: <root>/trossen_ee_world.hdf5 or current dir.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of episode indices to process (default: all).",
    )
    args = parser.parse_args()

    if LeRobotDataset is None:
        raise ImportError("lerobot is required. Install with: pip install lerobot")

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        episodes=args.episodes,
    )
    print(f"Loaded dataset: {len(dataset)} frames, {dataset.num_episodes} episodes.")

    urdf_path = args.urdf or _default_urdf_path()
    package_root = args.package_root or _default_package_root()
    if not Path(urdf_path).exists():
        print(f"Warning: URDF not found at {urdf_path}. You must pass a valid --urdf.")

    if args.output:
        output_path = Path(args.output)
    else:
        # Default: same directory as dataset (local root or HF cache)
        root = args.root or getattr(dataset.meta, "root", None)
        root = Path(root) if root else Path.cwd()
        output_path = root / "trossen_ee_world.hdf5"

    run_fk_and_save(
        dataset,
        urdf_path=urdf_path,
        package_root=package_root if Path(package_root).exists() else None,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
