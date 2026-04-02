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
Utilities to control a robot.

Useful to record a dataset, replay a recorded episode, run the policy on your robot
and record an evaluation dataset, and to recalibrate your robot if needed.

Examples of usage:

- Recalibrate your robot:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=calibrate
```

- Unlimited teleoperation at highest frequency (~200 Hz is expected), to exit with CTRL+C:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --robot.cameras='{}' \
    --control.type=teleoperate

# Add the cameras from the robot definition to visualize them:
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate
```

- Unlimited teleoperation at a limited frequency of 30 Hz, to simulate data recording frequency:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate \
    --control.fps=30
```

- Record one episode in order to test replay:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/koch_test \
    --control.num_episodes=1 \
    --control.push_to_hub=True
```

- Visualize dataset:
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id $USER/koch_test \
    --episode-index 0
```

- Replay this test episode:
```bash
python lerobot/scripts/control_robot.py replay \
    --robot.type=so100 \
    --control.type=replay \
    --control.fps=30 \
    --control.repo_id=$USER/koch_test \
    --control.episode=0
```

- Dataset replay: run the policy conditioned on trajectory (left_ee_position, right_ee_position) from a dataset episode:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=trossen_ai_mobile \
    --control.type=dataset_replay \
    --control.repo_id=user/dataset_with_fk \
    --control.episode=0 \
    --control.policy.path=outputs/train/act_with_trajectory/checkpoints/080000/pretrained_model \
    --control.fps=30
```
(The dataset must contain left_ee_position and right_ee_position; the policy must accept trajectory conditioning.)

- Record a full dataset in order to train a policy, with 2 seconds of warmup,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python lerobot/scripts/control_robot.py record \
    --robot.type=so100 \
    --control.type=record \
    --control.fps 30 \
    --control.repo_id=$USER/koch_pick_place_lego \
    --control.num_episodes=50 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10
```

- For remote controlled robots like LeKiwi, run this script on the robot edge device (e.g. RaspBerryPi):
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=lekiwi \
  --control.type=remote_robot
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to resseting the environment.
- Tap right arrow key '->' to early exit while resetting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command and adding `--control.resume=true`.

- Train on this dataset with the ACT policy:
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/koch_pick_place_lego \
  --policy.type=act \
  --output_dir=outputs/train/act_koch_pick_place_lego \
  --job_name=act_koch_pick_place_lego \
  --device=cuda \
  --wandb.enable=true
```

- Run the pretrained policy on the robot:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/eval_act_koch_pick_place_lego \
    --control.num_episodes=10 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10 \
    --control.push_to_hub=true \
    --control.policy.path=outputs/train/act_koch_pick_place_lego/checkpoints/080000/pretrained_model
```
"""

import logging
import time
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import tqdm

import cv2
import numpy as np
import torch
import json

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    CalibrateControlConfig,
    ControlPipelineConfig,
    DatasetReplayControlConfig,
    RecordControlConfig,
    RemoteRobotConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
)
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    init_keyboard_listener,
    log_control_info,
    predict_action,
    record_episode,
    reset_environment,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,
)
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import has_method, init_logging, log_say
from lerobot.configs import parser

from data_processing.compute_fk_from_dataset import (
    _project_ee_to_image,
    _project_cam_to_image,
    _load_urdf,
    _build_state_to_q_mapping,
    compute_fk_for_frame,
    LEFT_EE_LINK,
    RIGHT_EE_LINK,
    HEAD_CAMERA_LINK,
    DEFAULT_FX,
    DEFAULT_FY,
    DEFAULT_CX,
    DEFAULT_CY,
)

from split_episodes import TASKS

########################################################################################
# Control modes
########################################################################################


@safe_disconnect
def calibrate(robot: Robot, cfg: CalibrateControlConfig):
    # TODO(aliberts): move this code in robots' classes
    if robot.robot_type.startswith("stretch"):
        if not robot.is_connected:
            robot.connect()
        if not robot.is_homed():
            robot.home()
        return

    arms = robot.available_arms if cfg.arms is None else cfg.arms
    unknown_arms = [arm_id for arm_id in arms if arm_id not in robot.available_arms]
    available_arms_str = " ".join(robot.available_arms)
    unknown_arms_str = " ".join(unknown_arms)

    if arms is None or len(arms) == 0:
        raise ValueError(
            "No arm provided. Use `--arms` as argument with one or more available arms.\n"
            f"For instance, to recalibrate all arms add: `--arms {available_arms_str}`"
        )

    if len(unknown_arms) > 0:
        raise ValueError(
            f"Unknown arms provided ('{unknown_arms_str}'). Available arms are `{available_arms_str}`."
        )

    for arm_id in arms:
        arm_calib_path = robot.calibration_dir / f"{arm_id}.json"
        if arm_calib_path.exists():
            print(f"Removing '{arm_calib_path}'")
            arm_calib_path.unlink()
        else:
            print(f"Calibration file not found '{arm_calib_path}'")

    if robot.is_connected:
        robot.disconnect()

    if robot.robot_type.startswith("lekiwi") and "main_follower" in arms:
        print("Calibrating only the lekiwi follower arm 'main_follower'...")
        robot.calibrate_follower()
        return

    if robot.robot_type.startswith("lekiwi") and "main_leader" in arms:
        print("Calibrating only the lekiwi leader arm 'main_leader'...")
        robot.calibrate_leader()
        return

    # Calling `connect` automatically runs calibration
    # when the calibration file is missing
    robot.connect()
    robot.disconnect()
    print("Calibration is done! You can now teleoperate and record datasets!")


@safe_disconnect
def teleoperate(robot: Robot, cfg: TeleoperateControlConfig):
    control_loop(
        robot,
        control_time_s=cfg.teleop_time_s,
        fps=cfg.fps,
        teleoperate=True,
        display_cameras=cfg.display_cameras,
    )


@safe_disconnect
def record(
    robot: Robot,
    cfg: RecordControlConfig,
) -> LeRobotDataset:
    # TODO(rcadene): Add option to record logs

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.repo_id,
            root=cfg.root,
        )
        if len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.num_image_writer_processes,
                num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.fps, cfg.video)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.root,
            robot=robot,
            use_videos=cfg.video,
            image_writer_processes=cfg.num_image_writer_processes,
            image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
        )

    # Load pretrained policy
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    # Disable the leader arms if a policy is provided,
    # as they are not used during evaluation.
    if policy is not None:
        robot.leader_arms = []

    if not robot.is_connected:
        robot.connect()

    listener, events = init_keyboard_listener()

    # Execute a few seconds without recording to:
    # 1. teleoperate the robot to move it in starting position if no policy provided,
    # 2. give times to the robot devices to connect and start synchronizing,
    # 3. place the cameras windows on screen
    enable_teleoperation = policy is None
    log_say("Warmup record", cfg.play_sounds)
    warmup_record(robot, events, enable_teleoperation, cfg.warmup_time_s, cfg.display_cameras, cfg.fps)

    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    recorded_episodes = 0
    try:
        while True:
            if recorded_episodes >= cfg.num_episodes:
                break

            if has_method(robot, "enable_teleoperation"):
                robot.enable_teleoperation()

            log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
            record_episode(
                robot=robot,
                dataset=dataset,
                events=events,
                episode_time_s=cfg.episode_time_s,
                display_cameras=cfg.display_cameras,
                policy=policy,
                fps=cfg.fps,
                single_task=cfg.single_task,
            )

            # Execute a few seconds without recording to give time to manually reset the environment
            # Current code logic doesn't allow to teleoperate during this time.
            # TODO(rcadene): add an option to enable teleoperation during reset
            # Skip reset for the last episode to be recorded
            if not events["stop_recording"] and (
                (recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment", cfg.play_sounds)
                reset_environment(robot, events, cfg.reset_time_s, cfg.fps)

            if events["rerecord_episode"]:
                log_say("Re-record episode", cfg.play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.add_episode_to_batch()

            recorded_episodes += 1

            if (
                cfg.save_interval > 0
                and cfg.save_interval != 0
                and recorded_episodes % cfg.save_interval == 0
                and recorded_episodes > 0
            ):
                log_say("Encoding and saving dataset batch...", cfg.play_sounds)
                dataset.save_episode_batch()

            if events["stop_recording"]:
                break

        log_say("Stop recording", cfg.play_sounds, blocking=True)
        stop_recording(robot, listener, cfg.display_cameras)

        if cfg.save_interval <= 0 or (recorded_episodes % cfg.save_interval != 0):
            log_say("Encoding and saving dataset batch...", cfg.play_sounds)
            dataset.save_episode_batch()

    except Exception as e:
        logging.error(f"An exception occurred: {e}", exc_info=True)
    finally:
        logging.info("Saving dataset...")
        try:
            dataset.save_episode_batch()
        except Exception as save_exc:
            logging.error(
                f"Exception occurred while saving dataset in finally block: {save_exc}", exc_info=True
            )

    if cfg.push_to_hub:
        dataset.push_to_hub(tags=cfg.tags, private=cfg.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset


@safe_disconnect
def replay(
    robot: Robot,
    cfg: ReplayControlConfig,
):
    # TODO(rcadene, aliberts): refactor with control_loop, once `dataset` is an instance of LeRobotDataset
    # TODO(rcadene): Add option to record logs

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root, episodes=[cfg.episode])
    actions = dataset.hf_dataset.select_columns("action")

    # Disable leader arms as they are not used during replay
    robot.leader_arms = []

    if not robot.is_connected:
        robot.connect()

    log_say("Replaying episode", cfg.play_sounds, blocking=True)
    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()

        action = actions[idx]["action"]
        robot.send_action(action)

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / cfg.fps - dt_s)

        dt_s = time.perf_counter() - start_episode_t
        log_control_info(robot, dt_s, fps=cfg.fps)


def _obs_tensor_to_numpy_bgr(tensor: torch.Tensor) -> np.ndarray:
    """Convert a robot observation image tensor (C,H,W float [0,1] or uint8) to BGR numpy for cv2."""
    img = tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype in (np.float32, np.float64):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(img)


def _draw_ee_trajectory_on_image(
    img: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    left_ee: np.ndarray,
    right_ee: np.ndarray,
) -> np.ndarray:
    """Draw projected EE trajectory points onto a BGR image.

    left_ee / right_ee can be (3,) for a single point or (N, 3) for a trajectory chunk.
    The first point in the chunk (current timestep) gets a large circle; subsequent
    points are drawn as a fading polyline showing the future trajectory.
    """
    img = img.copy()
    cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(-1)[:3]
    cam_quat = np.asarray(cam_quat, dtype=np.float64).reshape(-1)[:4]

    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    left_pts = _ensure_2d(left_ee)
    right_pts = _ensure_2d(right_ee)

    def project_single(pt: np.ndarray) -> tuple[int, int] | None:
        uv = _project_cam_to_image(
            pt, DEFAULT_FX, DEFAULT_FY, DEFAULT_CX, DEFAULT_CY,
        )
        return uv

    left_uvs = [project_single(left_pts[i]) for i in range(len(left_pts))]
    right_uvs = [project_single(right_pts[i]) for i in range(len(right_pts))]

    def draw_trajectory(uvs, color_bgr, label):
        valid = [(i, uv) for i, uv in enumerate(uvs) if uv is not None]
        if not valid:
            return
        # Draw polyline for future trajectory.
        if len(valid) > 1:
            line_pts = np.array([uv for _, uv in valid], dtype=np.int32)
            cv2.polylines(img, [line_pts], isClosed=False, color=color_bgr, thickness=2)
        # Current position: large filled circle.
        i0, uv0 = valid[0]
        cv2.circle(img, uv0, 5, color_bgr, -1)
        cv2.putText(img, label, (uv0[0] + 12, uv0[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
        # Future positions: small dots.
        for i, uv in valid[1:]:
            alpha = max(0.2, 1.0 - i / len(uvs))
            r = max(2, int(6 * alpha))
            cv2.circle(img, uv, r, color_bgr, -1)

    draw_trajectory(left_uvs, (0, 255, 0), "L")   # light blue in BGR (almost cyan)
    draw_trajectory(right_uvs, (0, 255, 255), "R")  # light red in BGR
    return img


def _draw_ee_trajectory_on_image_colored(
    img: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    left_ee: np.ndarray,
    right_ee: np.ndarray,
    left_color: tuple[int, int, int] = (255, 200, 100),
    right_color: tuple[int, int, int] = (100, 100, 255),
    left_label: str = "L_gt",
    right_label: str = "R_gt",
) -> np.ndarray:
    """Same as _draw_ee_trajectory_on_image but with configurable BGR colors and labels."""
    img = img.copy()
    cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(-1)[:3]
    cam_quat = np.asarray(cam_quat, dtype=np.float64).reshape(-1)[:4]

    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    left_pts = _ensure_2d(left_ee)
    right_pts = _ensure_2d(right_ee)

    def project_single(pt: np.ndarray) -> tuple[int, int] | None:
        uv, _ = _project_ee_to_image(
            cam_pos, cam_quat, pt, pt,
            DEFAULT_FX, DEFAULT_FY, DEFAULT_CX, DEFAULT_CY,
        )
        return uv

    left_uvs = [project_single(left_pts[i]) for i in range(len(left_pts))]
    right_uvs = [project_single(right_pts[i]) for i in range(len(right_pts))]

    def draw_trajectory(uvs, color_bgr, label):
        valid = [(i, uv) for i, uv in enumerate(uvs) if uv is not None]
        if not valid:
            return
        if len(valid) > 1:
            line_pts = np.array([uv for _, uv in valid], dtype=np.int32)
            cv2.polylines(img, [line_pts], isClosed=False, color=color_bgr, thickness=2)
        i0, uv0 = valid[0]
        cv2.circle(img, uv0, 5, color_bgr, -1)
        cv2.putText(img, label, (uv0[0] + 12, uv0[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
        for i, uv in valid[1:]:
            alpha = max(0.2, 1.0 - i / len(uvs))
            r = max(2, int(6 * alpha))
            cv2.circle(img, uv, r, color_bgr, -1)

    draw_trajectory(left_uvs, left_color, left_label)
    draw_trajectory(right_uvs, right_color, right_label)
    return img


def _save_video_cv2(path: str, frames: list[np.ndarray], fps: int):
    """Save a list of BGR numpy frames as an mp4 video."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    logging.info(f"Wrote {len(frames)} frames to {path}")


class MolmoEEPredictor:
    """Predicts left/right EE positions using a Molmo VLA checkpoint.

    Wraps model loading, preprocessing, and flow-matching inference following
    the pattern in eval_closed_loop.py.  Output is split into left_ee_position
    (action_horizon, 3) and right_ee_position (action_horizon, 3).
    """

    def __init__(self, checkpoint: str, device: str = "cuda", num_ode_steps: int = 10):
        from olmo.model import Molmo
        from olmo.data import build_mm_preprocessor

        logging.info(f"Loading Molmo model from {checkpoint}")
        self.model = Molmo.from_checkpoint(checkpoint, device=device)
        self.model.eval()
        self.preprocessor = build_mm_preprocessor(
            self.model.config, for_inference=True, is_training=False
        )
        self.device = torch.device(device)
        self.num_ode_steps = num_ode_steps
        self.action_horizon = self.model.config.action_horizon
        self.action_dim = self.model.config.action_dim

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        proprio_state: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Molmo and return (left_ee, right_ee) each of shape (action_horizon, 3).

        Args:
            image: RGB uint8 array (H, W, 3) from the head camera.
            instruction: Language instruction / task description.
            proprio_state: Optional proprioceptive state vector.

        Returns:
            (left_ee_position, right_ee_position) as float32 numpy arrays.
        """
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image)
        example = {
            "image": pil_image,
            "prompt": instruction,
            "style": "trajectory_3d_egodex_trossen_direct",
            "proprio_state": proprio_state,
        }
        if proprio_state is not None:
            example["state"] = proprio_state

        batch = self.preprocessor(example)

        input_ids = torch.tensor(batch["input_tokens"], dtype=torch.long).unsqueeze(0).to(self.device)
        images = torch.tensor(batch["images"], dtype=torch.float32).unsqueeze(0).to(self.device)
        image_input_idx = torch.tensor(batch["image_input_idx"], dtype=torch.long).unsqueeze(0).to(self.device)

        image_masks = None
        if "image_masks" in batch:
            image_masks = torch.tensor(batch["image_masks"]).unsqueeze(0).to(self.device)
        position_ids = None
        if "position_ids" in batch:
            position_ids = torch.tensor(batch["position_ids"], dtype=torch.long).unsqueeze(0).to(self.device)
        proprio_tensor = None
        if "proprio_state" in batch:
            proprio_tensor = torch.tensor(batch["proprio_state"], dtype=torch.float32).unsqueeze(0).to(self.device)

        expert_type = torch.tensor([1], dtype=torch.long).to(self.device)
        initial_noise = torch.randn(1, self.action_horizon, self.action_dim, device=self.device)

        # actions = self.model.predict_actions_direct(
        #             input_ids=input_ids,
        #             attention_mask=None,
        #             images=images,
        #             image_masks=image_masks,
        #             image_input_idx=image_input_idx,
        #             position_ids=position_ids,
        #             proprio_state=proprio_tensor,
        #             expert_type=expert_type,
        #         )

        actions = self.model.sample_actions_flow_matching(
            input_ids=input_ids,
            attention_mask=None,
            images=images,
            image_masks=image_masks,
            image_input_idx=image_input_idx,
            num_steps=self.num_ode_steps,
            initial_noise=initial_noise,
            position_ids=position_ids,
            proprio_state=proprio_tensor,
            expert_type=expert_type,
        )
        if isinstance(actions, tuple):
            actions = actions[0]

        actions_np = actions.cpu().numpy()[0]  # (action_horizon, action_dim)

        # # First 3 dims = left EE xyz, next 3 = right EE xyz.
        # left_ee = actions_np[:, :3].astype(np.float32)
        # right_ee = actions_np[:, 3:6].astype(np.float32)
        # return left_ee, right_ee
        return actions_np


@safe_disconnect
def dataset_replay(
    robot: Robot,
    cfg: DatasetReplayControlConfig,
):
    """Run the policy on the robot conditioned on trajectory (left_ee_position, right_ee_position) from a dataset episode.

    Loads one episode from the LeRobot dataset (must have left_ee_position and right_ee_position).
    At each step: get current robot observation, get trajectory chunk from dataset, run policy, execute action.
    """
    if cfg.policy is None:
        raise ValueError(
            "dataset_replay requires a policy. Pass --control.policy.path=/path/to/checkpoint"
        )

    # Load dataset and policy BEFORE connecting the robot so the base hardware
    # doesn't time out while we download data / load model weights.
    ds_meta = LeRobotDatasetMetadata(cfg.repo_id, root=cfg.root)
    delta_timestamps, trajectory_random_window = resolve_delta_timestamps(
        cfg.policy, ds_meta, use_trajectory_random_window=True
    )
    dataset = LeRobotDataset(
        cfg.repo_id,
        root=cfg.root,
        delta_timestamps=delta_timestamps,
        trajectory_random_window=trajectory_random_window,
    )
    episode_start = dataset.episode_data_index["from"][cfg.episode].item()
    episode_end = dataset.episode_data_index["to"][cfg.episode].item()

    if cfg.molmo_checkpoint is None:
        if "left_ee_position" not in dataset.features or "right_ee_position" not in dataset.features:
            raise ValueError(
                "dataset_replay requires a dataset with left_ee_position and right_ee_position "
                "(or --control.molmo_checkpoint to predict them). "
                f"Got features: {list(dataset.features.keys())}"
            )

    policy = make_policy(cfg.policy, ds_meta=dataset.meta)
    policy.reset()
    device = get_safe_torch_device(policy.config.device)
    fps = cfg.fps if cfg.fps is not None else dataset.fps

    # --------------- FK setup: compute EE positions from observation.state ---------------
    urdf_path = "/root/lerobot/data_processing/stationary_ai.urdf"
    fk_model = _load_urdf(urdf_path)
    fk_data = fk_model.createData()

    stats_path =f"/root/lerobot/data_processing/trajectory_stats.json"
    stats = json.load(open(stats_path))
    mean = torch.tensor(stats["trajectory_stats_mean"]).to(device)
    std = torch.tensor(stats["trajectory_stats_std"]).to(device)

    state_key = "observation.state"
    state_names = dataset.features[state_key].get("names")
    state_to_q = _build_state_to_q_mapping(fk_model, state_names)
    fk_frame_ids = {
        name: fk_model.getFrameId(name)
        for name in [LEFT_EE_LINK, RIGHT_EE_LINK, HEAD_CAMERA_LINK]
    }

    def _state_to_ee(state_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Run FK on an observation.state tensor and return (left_ee, right_ee) as float32 (3,)."""
        s = state_tensor.cpu().numpy() if isinstance(state_tensor, torch.Tensor) else np.asarray(state_tensor)
        s = np.asarray(s, dtype=np.float64).ravel()
        fk_out = compute_fk_for_frame(fk_model, fk_data, s, state_names, state_to_q, fk_frame_ids)
        
        return fk_out["left_ee_position"].astype(np.float32), fk_out["right_ee_position"].astype(np.float32)

    def _world_to_camera(
        p_world: np.ndarray,
        cam_position: np.ndarray,
        cam_quat_xyzw: np.ndarray,
    ) -> np.ndarray:
        """Transform world-frame point to camera frame. p_world (3,), returns (3,)."""
        from scipy.spatial.transform import Rotation
        R_world_to_cam = Rotation.from_quat(cam_quat_xyzw).as_matrix().T
        p_cam = R_world_to_cam @ (np.asarray(p_world).reshape(3) - np.asarray(cam_position).reshape(3))
        return p_cam

    def _interp1d_torch(vec, target_len=100):
        # vec: (N, C) e.g. (30, 3) -> treat as (1, C, N), interpolate to (1, C, target_len), -> (target_len, C)
        return torch.nn.functional.interpolate(
            vec.T.unsqueeze(0).float(),  # (1, C, N)
            size=target_len,
            mode='linear',
            align_corners=True,
        )[0].T  # (target_len, C)

    # Load Molmo model for EE prediction if checkpoint provided.
    molmo_predictor = None
    if cfg.molmo_checkpoint is not None:
        molmo_predictor = MolmoEEPredictor(
            checkpoint=str(cfg.molmo_checkpoint),
            device=str(device),
            num_ode_steps=cfg.molmo_num_ode_steps,
        )
        logging.info("Using Molmo model for EE position prediction")

    # # # Connect robot AFTER loading all models to minimize idle time.
    # if not robot.is_connected:
    #     robot.connect()

    # Video recording setup: collect annotated frames per camera.
    record_video = cfg.video_output_dir is not None
    video_frames: dict[str, list[np.ndarray]] = {}
    has_camera_pose = (
        "head_camera_position" in dataset.features
        and "head_camera_quat_xyzw" in dataset.features
    )
    cam_high_key = "observation.images.cam_high"

    # log_say("Dataset replay: running policy conditioned on trajectory", cfg.play_sounds, blocking=True)
    # for idx in range(episode_start, episode_end):
    for idx in tqdm.tqdm(range(0, 600), desc="Dataset replay"):
        start_t = time.perf_counter()

        # if idx % 100 != 0:
        #     continue

        frame = dataset[episode_start + idx]
        if robot.is_connected:
            robot_obs = robot.capture_observation()
        else:
            robot_obs = {
                "observation.state": frame["observation.state"],
                "observation.images.cam_high": frame["observation.images.cam_high"].permute(1, 2, 0),
                "observation.images.cam_low": frame["observation.images.cam_low"].permute(1, 2, 0),
                'observation.images.cam_left_wrist': frame['observation.images.cam_left_wrist'].permute(1, 2, 0),
                'observation.images.cam_right_wrist': frame['observation.images.cam_right_wrist'].permute(1, 2, 0),
            }

        batch = dict(robot_obs)

        # Assign task instructions uniformly as idx increases through total frames
        num_tasks = len(TASKS)
        # Uniform split: assign first N/num_tasks frames to task 0, next N/num_tasks to task 1, etc.
        frames_per_task = (episode_end - episode_start) // num_tasks
        task_idx = min(idx // frames_per_task, num_tasks - 1)  # Ensure doesn't go out of range for last task
        task_instruction = TASKS[task_idx]
        print(f"Task instruction: {task_idx} / {num_tasks}: {task_instruction}")

        if molmo_predictor is not None and (idx % 50 == 0 or idx == episode_start):
            # Predict EE positions from the live camera image using Molmo.
            cam_img = robot_obs.get(cam_high_key)
            if cam_img is None:
                cam_img = next(v for k, v in robot_obs.items() if "image" in k)
            img_np = cam_img.cpu().numpy() if isinstance(cam_img, torch.Tensor) else np.asarray(cam_img)
            if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
                img_np = np.transpose(img_np, (1, 2, 0))
            if img_np.dtype in (np.float32, np.float64):
                img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

            left_ee_curr, right_ee_curr = _state_to_ee(frame["observation.state"])
            # import ipdb; ipdb.set_trace()
            left_ee_curr = _world_to_camera(left_ee_curr, frame["head_camera_position"], frame["head_camera_quat_xyzw"])
            right_ee_curr = _world_to_camera(right_ee_curr, frame["head_camera_position"], frame["head_camera_quat_xyzw"])
            
            initial_ee = np.concatenate([left_ee_curr, right_ee_curr])

            # left_ee_curr = torch.from_numpy(left_ee_curr).to(device)
            # right_ee_curr = torch.from_numpy(right_ee_curr).to(device)

            
            traj_np = molmo_predictor.predict(
                image=img_np,
                instruction=task_instruction,
                proprio_state=initial_ee,
            )
            traj_flat = torch.from_numpy(traj_np).to(device)  # (T, 6)
            initial_ee = torch.from_numpy(initial_ee).to(device)

            traj_flat = traj_flat * std + mean
            traj_flat = torch.cumsum(traj_flat, dim=0) + initial_ee.unsqueeze(0)
            recovered = torch.cat([initial_ee.unsqueeze(0), traj_flat], dim=0)  # (T+1, 6)

            batch["left_ee_position"] = _interp1d_torch(recovered[:, :3], 100)
            batch["right_ee_position"] = _interp1d_torch(recovered[:, 3:6], 100)

            if idx % 50 == 0 or idx == episode_start:
                fk_out = compute_fk_for_frame(
                    fk_model, fk_data,
                    np.asarray(frame["observation.state"].cpu().numpy() if isinstance(frame["observation.state"], torch.Tensor) else frame["observation.state"], dtype=np.float64).ravel(),
                    state_names, state_to_q, fk_frame_ids,
                )
                cam_pos = fk_out["head_camera_position"]
                cam_quat = fk_out["head_camera_quat_xyzw"]

                # Molmo-predicted (interpolated) EE trajectory
                left_vis = recovered[:, :3].cpu().numpy()
                
                right_vis = recovered[:, 3:6].cpu().numpy()

                # Ground-truth EE trajectory from dataset
                gt_left = frame["left_ee_position"]
                gt_right = frame["right_ee_position"]
                if isinstance(gt_left, torch.Tensor):
                    gt_left = gt_left.cpu().numpy()
                if isinstance(gt_right, torch.Tensor):
                    gt_right = gt_right.cpu().numpy()

                vis_img = img_np.copy()

                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                # Write task_instruction string text onto vis_img (cv2 image) at the top left starting from (10, 30).
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                text_color = (255, 255, 255)  # white
                outline_color = (0, 0, 0)    # black for contrast
                x, y = 10, 30  # Top-left starting coordinates for text baseline

                # To make the text more readable, first draw the text outline (thicker, black)
                cv2.putText(
                    vis_img,
                    f"{task_instruction}",
                    (x, y),
                    font,
                    font_scale,
                    outline_color,
                    font_thickness + 2,
                    lineType=cv2.LINE_AA
                )
                # Draw the actual text (white)
                cv2.putText(
                    vis_img,
                    f"{task_instruction}",
                    (x, y),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                    lineType=cv2.LINE_AA
                )
                
                # Draw ground-truth: green (L) / yellow (R)
                vis_img = _draw_ee_trajectory_on_image_colored(
                    vis_img, cam_pos, cam_quat, gt_left, gt_right,
                    # left_color=(0, 255, 0), right_color=(0, 255, 255),
                    left_label="L_gt", right_label="R_gt",
                )
                # Draw Molmo predictions: blue (L) / red (R)
                vis_img = _draw_ee_trajectory_on_image(vis_img, cam_pos, cam_quat, left_vis, right_vis)

                save_dir = Path(cfg.video_output_dir or "ee_debug_frames")
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"frame_{idx:06d}.png"
                cv2.imwrite(str(save_path), vis_img)
                logging.info(f"Saved EE overlay frame to {save_path}")
        else:
            batch["left_ee_position"] = frame["left_ee_position"]
            batch["right_ee_position"] = frame["right_ee_position"]


        # import ipdb; ipdb.set_trace()

        action = predict_action(batch, policy, device, policy.config.use_amp)

        if robot.is_connected:
            robot.send_action(action)
        

        # # Resolve EE values for video overlay (use whichever source we used above).
        # if molmo_predictor is not None and idx % 100 == 0:
        #     left_ee_for_vis = recovered[:, :3].cpu().numpy()
        #     right_ee_for_vis = recovered[:, 3:6].cpu().numpy()
        # else:
        #     left_ee_for_vis = frame["left_ee_position"]
        #     right_ee_for_vis = frame["right_ee_position"]
            
        # if isinstance(left_ee_for_vis, torch.Tensor):
        #     left_ee_for_vis = left_ee_for_vis.cpu().numpy()
        # if isinstance(right_ee_for_vis, torch.Tensor):
        #     right_ee_for_vis = right_ee_for_vis.cpu().numpy()

        if record_video:
            for cam_key in robot_obs:
                if "image" not in cam_key:
                    continue
                img_tensor = robot_obs[cam_key]
                img = _obs_tensor_to_numpy_bgr(img_tensor)

                if cam_key == cam_high_key and has_camera_pose:
                    cam_pos = frame["head_camera_position"]
                    cam_quat = frame["head_camera_quat_xyzw"]
                    if isinstance(cam_pos, torch.Tensor):
                        cam_pos = cam_pos.cpu().numpy()
                    if isinstance(cam_quat, torch.Tensor):
                        cam_quat = cam_quat.cpu().numpy()

                    # Ground-truth EE trajectory from dataset
                    gt_left = frame["left_ee_position"]
                    gt_right = frame["right_ee_position"]
                    if isinstance(gt_left, torch.Tensor):
                        gt_left = gt_left.cpu().numpy()
                    if isinstance(gt_right, torch.Tensor):
                        gt_right = gt_right.cpu().numpy()
                    img = _draw_ee_trajectory_on_image_colored(
                        img, cam_pos, cam_quat, gt_left, gt_right,
                        # left_color=(0, 255, 0), right_color=(0, 255, 255),
                        left_label="L_gt", right_label="R_gt",
                    )
                    if molmo_predictor is not None:
                        img = _draw_ee_trajectory_on_image(
                            img, cam_pos, cam_quat, left_vis, right_vis,
                        )
                # Add text overlay displaying the current step on the video frame
                step_text = f"Step: {idx+1}/{episode_end - episode_start}: {task_instruction}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_color = (255, 255, 255)
                thickness = 2
                margin = 10
                # Put the text at the top-left corner of the image
                cv2.putText(
                    img,
                    step_text,
                    (margin, margin + 25),
                    font,
                    font_scale,
                    font_color,
                    thickness,
                    lineType=cv2.LINE_AA
                )
                video_frames.setdefault(cam_key, []).append(img)

        dt_s = time.perf_counter() - start_t
        busy_wait(1 / fps - dt_s)
        log_control_info(robot, dt_s, fps=fps)

    policy.reset()

    if record_video and video_frames:
        out_dir = Path(cfg.video_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for cam_key, frames_list in video_frames.items():
            safe_name = cam_key.replace(".", "_").replace("/", "_")
            video_path = out_dir / f"rollout_ep{cfg.episode}_{safe_name}.mp4"
            _save_video_cv2(str(video_path), frames_list, fps)
            logging.info(f"Saved rollout video to {video_path}")


@parser.wrap()
def control_robot(cfg: ControlPipelineConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    

    if isinstance(cfg.control, CalibrateControlConfig):
        calibrate(robot, cfg.control)
    elif isinstance(cfg.control, TeleoperateControlConfig):
        teleoperate(robot, cfg.control)
    elif isinstance(cfg.control, RecordControlConfig):
        record(robot, cfg.control)
    elif isinstance(cfg.control, ReplayControlConfig):
        replay(robot, cfg.control)
    elif isinstance(cfg.control, DatasetReplayControlConfig):
        dataset_replay(robot, cfg.control)
    elif isinstance(cfg.control, RemoteRobotConfig):
        from lerobot.common.robot_devices.robots.lekiwi_remote import run_lekiwi

        run_lekiwi(cfg.robot)

    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()


if __name__ == "__main__":
    control_robot()
