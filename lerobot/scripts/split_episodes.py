#!/usr/bin/env python
"""
Split each episode in a LeRobot dataset into sub-episodes by task, using a
two-pass VLM pipeline (GPT-4o) to detect task boundaries from sampled frames.

Pass 1: Classify each sampled frame individually using all camera views.
Pass 2: Densely re-sample around detected transition points to pinpoint
         exact task boundaries.

Required:
    export OPENAI_API_KEY=sk-...

Usage:
    python lerobot/scripts/split_episodes.py \
        --repo-id user/source_dataset \
        --input-dir /path/to/source_dataset \
        --output-dir /path/to/output_dataset \
        [--sample-every 10] \
        [--model gpt-4o] \
        [--camera-key cam_high,cam_left_wrist] \
        [--push-repo-id user/new_dataset]
"""

import argparse
import base64
import io
import json
import logging
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import jsonlines
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from openai import OpenAI
from PIL import Image

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_episode_data_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TASKS = [
    "Stir the pot",
    "Unzip the pencil case",
    "Uncap the red marker",
    "Open the red drawer",
    "Uncap the bottle",
]

TASK_TO_INDEX = {t: i for i, t in enumerate(TASKS)}

DEFAULT_CHUNK_SIZE = 1000

SYSTEM_MSG = (
    "You are a computer-vision assistant for a robotics research lab. "
    "You analyze image frames from a robot arm performing household tasks. "
    "You must determine the task SOLELY from what you see in the image — "
    "which object the gripper is interacting with, how it moves, and what "
    "is visible on the table. Never guess or assume a default order. "
    "Always respond with ONLY the exact task name, nothing else."
)

TASKS_BULLET = "\n".join(f"  - {t}" for t in TASKS)

# ── I/O helpers ──────────────────────────────────────────────────────────────


def load_json(fpath: Path) -> dict:
    with open(fpath) as f:
        return json.load(f)


def write_json(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_jsonlines(fpath: Path) -> list:
    with jsonlines.open(fpath, "r") as reader:
        return list(reader)


def write_jsonlines(data: list, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(fpath, "w") as writer:
        writer.write_all(data)


def get_episode_chunk(ep_index: int, chunks_size: int = DEFAULT_CHUNK_SIZE) -> int:
    return ep_index // chunks_size


def get_data_file_path(data_path_template: str, ep_index: int, chunks_size: int) -> Path:
    ep_chunk = get_episode_chunk(ep_index, chunks_size)
    return Path(data_path_template.format(episode_chunk=ep_chunk, episode_index=ep_index))


def get_video_file_path(video_path_template: str, ep_index: int, vid_key: str, chunks_size: int) -> Path:
    ep_chunk = get_episode_chunk(ep_index, chunks_size)
    return Path(video_path_template.format(episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index))


def pil_to_base64(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _item_to_numpy(img) -> np.ndarray | None:
    """Convert a LeRobotDataset image item (tensor/ndarray/PIL) to uint8 HWC numpy."""
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.dtype in (np.float32, np.float64):
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return img
    if isinstance(img, Image.Image):
        return np.array(img)
    return None


# ── Multi-camera frame composition ──────────────────────────────────────────


def compose_multi_camera_frame(
    dataset: LeRobotDataset,
    episode_index: int,
    frame_idx: int,
    camera_keys: list[str],
) -> Image.Image:
    """Load one frame from all cameras and tile them side-by-side."""
    ep_data_index = get_episode_data_index(dataset.meta.episodes, [episode_index])
    ep_start = ep_data_index["from"][0].item()
    global_idx = ep_start + frame_idx
    item = dataset[global_idx]

    images = []
    for cam_key in camera_keys:
        arr = _item_to_numpy(item[cam_key])
        if arr is not None:
            images.append(Image.fromarray(arr))

    if not images:
        raise RuntimeError(f"No camera images decoded for ep {episode_index} frame {frame_idx}")

    total_w = sum(im.width for im in images)
    max_h = max(im.height for im in images)
    composite = Image.new("RGB", (total_w, max_h))
    x_offset = 0
    for im in images:
        composite.paste(im, (x_offset, 0))
        x_offset += im.width

    return composite


# ── Single-frame VLM classification ─────────────────────────────────────────


def classify_single_frame(
    pil_image: Image.Image,
    client: OpenAI,
    model: str,
) -> str | None:
    """Send one composed multi-camera image to VLM, return the task label."""
    img_b64 = pil_to_base64(pil_image)

    prompt = (
        "This image shows a robot arm from one or more camera angles, "
        "performing one of the following household tasks:\n"
        f"{TASKS_BULLET}\n\n"
        "Look at what object the robot gripper is interacting with "
        "(a pot, a pencil case, a red marker, a red drawer, or a bottle) "
        "and reply with ONLY the exact task name from the list above. "
        "Do not add any explanation."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=50,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("  VLM call failed: %s", e)
        return None

    # Match against known tasks (case-insensitive, allows partial match)
    for task in TASKS:
        if task.lower() in raw.lower():
            return task

    logger.warning("  VLM returned unrecognised label: '%s'", raw)
    return None


# ── Pass 1: Coarse per-frame classification ─────────────────────────────────


def classify_frames_pass1(
    dataset: LeRobotDataset,
    episode_index: int,
    camera_keys: list[str],
    frame_indices: list[int],
    client: OpenAI,
    model: str,
) -> list[tuple[int, str | None]]:
    """Classify every sampled frame. Returns [(frame_idx, task_label_or_None)]."""
    results = []
    total = len(frame_indices)
    for i, fi in enumerate(frame_indices):
        logger.info("    Pass 1: frame %d  (%d/%d)", fi, i + 1, total)
        composite = compose_multi_camera_frame(dataset, episode_index, fi, camera_keys)
        label = classify_single_frame(composite, client, model)
        results.append((fi, label))
        if label:
            logger.info("      → %s", label)
        else:
            logger.warning("      → UNKNOWN")
    return results


# ── Chunk-based boundary detection ───────────────────────────────────────────


def find_task_segments(
    coarse_labels: list[tuple[int, str | None]],
    total_frames: int,
    sample_every: int,
) -> list[dict] | None:
    """
    Find the 5 task segments from noisy per-frame labels.

    Key insight: during the "approach" phase (robot moving towards the next
    object) predictions are noisy.  During the "perform" phase (actually
    doing the task) the same label appears in a long contiguous run.

    Algorithm:
      1. Build contiguous runs of identical labels (ignoring None).
      2. For each unique task, pick its longest run — that is the "perform"
         window where the VLM was confident.
      3. Sort the 5 chosen runs chronologically.
      4. Split right after each task's perform window ends.  Each sub-episode
         includes the preceding approach frames plus the task performance.
    """
    # --- 1. Build contiguous runs, tolerating up to max_gap stray mispredictions ---
    max_gap = 2  # allow up to this many wrong predictions inside a run

    clean = [(fi, lbl) for fi, lbl in coarse_labels if lbl is not None]
    if len(clean) < 2:
        logger.error("  Too few classified frames (%d) to find task chunks.", len(clean))
        return None

    # First pass: strict contiguous runs
    raw_runs: list[dict] = []
    cur_label = clean[0][1]
    cur_start = clean[0][0]
    cur_count = 1

    for i in range(1, len(clean)):
        fi, lbl = clean[i]
        if lbl == cur_label:
            cur_count += 1
        else:
            raw_runs.append({
                "label": cur_label,
                "start_fi": cur_start,
                "end_fi": clean[i - 1][0],
                "count": cur_count,
            })
            cur_label = lbl
            cur_start = fi
            cur_count = 1
    raw_runs.append({
        "label": cur_label,
        "start_fi": cur_start,
        "end_fi": clean[-1][0],
        "count": cur_count,
    })

    # Second pass: merge runs of the same label separated by <= max_gap different ones
    runs: list[dict] = [raw_runs[0]]
    for r in raw_runs[1:]:
        prev = runs[-1]
        if r["label"] == prev["label"]:
            # Same label — always merge
            prev["end_fi"] = r["end_fi"]
            prev["count"] += r["count"]
        else:
            # Check if this is a small stray blip between two runs of the same label
            # Look ahead: is there a same-label run within max_gap positions?
            runs.append(r)

    # Now merge across small gaps: if run[i] and run[i+2] have the same label
    # and run[i+1] is short (<= max_gap), absorb run[i+1] into run[i]
    merged = True
    while merged:
        merged = False
        new_runs = []
        i = 0
        while i < len(runs):
            if (
                i + 2 < len(runs)
                and runs[i]["label"] == runs[i + 2]["label"]
                and runs[i + 1]["count"] <= max_gap
            ):
                new_runs.append({
                    "label": runs[i]["label"],
                    "start_fi": runs[i]["start_fi"],
                    "end_fi": runs[i + 2]["end_fi"],
                    "count": runs[i]["count"] + runs[i + 1]["count"] + runs[i + 2]["count"],
                })
                i += 3
                merged = True
            else:
                new_runs.append(runs[i])
                i += 1
        runs = new_runs

    logger.info("  Found %d contiguous runs across %d labelled frames.", len(runs), len(clean))
    for r in runs:
        logger.info(
            "    run: '%s'  sampled frames %d-%d  (%d consecutive hits)",
            r["label"], r["start_fi"], r["end_fi"], r["count"],
        )

    # --- 2. For each task, pick its longest run ---
    best_run: dict[str, dict] = {}
    for r in runs:
        lbl = r["label"]
        if lbl not in best_run or r["count"] > best_run[lbl]["count"]:
            best_run[lbl] = r

    found_tasks = list(best_run.keys())
    logger.info("  Identified %d task(s) via longest runs: %s", len(found_tasks), found_tasks)

    if len(found_tasks) != len(TASKS):
        missing = [t for t in TASKS if t not in found_tasks]
        logger.warning(
            "  Expected %d tasks, found %d.  Missing: %s",
            len(TASKS), len(found_tasks), missing,
        )
        if len(found_tasks) < 2:
            return None

    # --- 3. Sort chosen runs chronologically ---
    chosen = sorted(best_run.values(), key=lambda r: r["start_fi"])

    # --- 4. Build segments: split right after each task's perform window ---
    # Place the boundary at 1/10th of the gap after the previous task's
    # perform window, so ~90% of the approach phase belongs to the upcoming
    # task's sub-episode.
    segments = []
    for i, run in enumerate(chosen):
        if i == 0:
            seg_start = 0
        else:
            prev_run = chosen[i - 1]
            gap = run["start_fi"] - prev_run["end_fi"]
            seg_start = prev_run["end_fi"] + max(1, gap // 10)

        if i == len(chosen) - 1:
            seg_end = total_frames - 1
        else:
            next_run = chosen[i + 1]
            gap = next_run["start_fi"] - run["end_fi"]
            seg_end = run["end_fi"] + max(0, gap // 10) - 1
            seg_end = max(seg_end, run["end_fi"])

        segments.append({
            "task": run["label"],
            "start_frame": seg_start,
            "end_frame": seg_end,
        })

    segments.sort(key=lambda s: s["start_frame"])
    logger.info("  Final segments: %s", json.dumps(segments, indent=2))
    return segments


# ── Orchestrator ─────────────────────────────────────────────────────────────


def detect_task_boundaries_vlm(
    dataset: LeRobotDataset,
    episode_index: int,
    total_frames: int,
    fps: int,
    camera_keys: list[str],
    sample_every: int,
    model: str,
    client: OpenAI,
) -> list[dict] | None:
    """
    Per-frame VLM classification + chunk-based boundary detection.

    1. Classify every sampled frame individually (all cameras, full resolution).
    2. Find the longest contiguous run of each task label — that's the
       "performing" window where the robot is actually doing the task.
    3. Split right after each task's performance window ends, so each
       sub-episode = approach + task performance.

    Returns [{"task", "start_frame", "end_frame"}] sorted by start_frame, or None.
    """
    frame_indices = list(range(0, total_frames, sample_every))
    if frame_indices[-1] != total_frames - 1:
        frame_indices.append(total_frames - 1)

    logger.info(
        "  Classifying %d frames (every %d) for episode %d",
        len(frame_indices), sample_every, episode_index,
    )

    coarse_labels = classify_frames_pass1(
        dataset, episode_index, camera_keys, frame_indices, client, model,
    )

    labelled_count = sum(1 for _, lbl in coarse_labels if lbl is not None)
    logger.info(
        "  Classification done: %d/%d frames labelled successfully.",
        labelled_count, len(coarse_labels),
    )

    if labelled_count < 2:
        logger.error("  Not enough frames labelled — skipping episode %d.", episode_index)
        return None

    logger.info("  Finding task chunks for episode %d ...", episode_index)
    segments = find_task_segments(coarse_labels, total_frames, sample_every)

    return segments


# ── Review phase: re-predict task label on each split ────────────────────────


def review_segment_label(
    dataset: LeRobotDataset,
    episode_index: int,
    start_frame: int,
    end_frame: int,
    camera_keys: list[str],
    client: OpenAI,
    model: str,
    num_review_samples: int = 5,
) -> str | None:
    """
    Sample frames from a sub-episode segment, classify each, and return
    the majority-vote task label.
    """
    seg_len = end_frame - start_frame + 1
    if seg_len <= 0:
        return None

    # Sample frames from the second half of the segment (where the task
    # is most likely being performed, not the approach phase)
    half_start = start_frame + seg_len // 2
    step = max(1, (end_frame - half_start) // num_review_samples)
    review_indices = list(range(half_start, end_frame + 1, step))[-num_review_samples:]

    votes: dict[str, int] = {}
    for fi in review_indices:
        composite = compose_multi_camera_frame(dataset, episode_index, fi, camera_keys)
        lbl = classify_single_frame(composite, client, model)
        if lbl:
            votes[lbl] = votes.get(lbl, 0) + 1

    if not votes:
        return None

    winner = max(votes, key=votes.get)
    logger.info(
        "    Review votes: %s -> '%s'", dict(votes), winner,
    )
    return winner


# ── Video splitting ──────────────────────────────────────────────────────────


def split_video(input_path: Path, output_path: Path, start_frame: int, num_frames: int, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = start_frame / fps
    duration = num_frames / fps
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.6f}",
        "-i", str(input_path),
        "-t", f"{duration:.6f}",
        "-c:v", "libx264",
        "-an",
        "-loglevel", "error",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


# ── Per-sub-episode stats ────────────────────────────────────────────────────


def compute_sub_episode_stats(table: pa.Table, features: dict) -> dict:
    stats = {}
    for col_name in table.column_names:
        if col_name not in features:
            continue
        ft = features[col_name]
        if ft["dtype"] in ("string", "image", "video"):
            continue
        try:
            col = table.column(col_name)
            if pa.types.is_list(col.type) or pa.types.is_fixed_size_list(col.type):
                arr = np.stack(col.to_pylist())
            else:
                arr = col.to_numpy()

            if arr.dtype == object:
                arr = np.stack(arr)

            arr = arr.astype(np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)

            stats[col_name] = {
                "min": np.min(arr, axis=0).tolist(),
                "max": np.max(arr, axis=0).tolist(),
                "mean": np.mean(arr, axis=0).tolist(),
                "std": np.std(arr, axis=0).tolist(),
                "count": [len(arr)],
            }
        except Exception as e:
            logger.warning("Skipping stats for column '%s': %s", col_name, e)
    return stats


# ── Push to hub ──────────────────────────────────────────────────────────────


def push_to_hub(output_dir: Path, repo_id: str) -> None:
    """Upload the newly created dataset to the Hugging Face Hub."""
    from huggingface_hub import HfApi

    logger.info("Pushing dataset to hub: %s", repo_id)
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(output_dir),
        repo_type="dataset",
    )
    logger.info("Dataset pushed to https://huggingface.co/datasets/%s", repo_id)


# ── Main ─────────────────────────────────────────────────────────────────────


def main(
    repo_id: str,
    input_dir: Path,
    output_dir: Path,
    sample_every: int,
    model: str,
    camera_keys_override: list[str] | None,
    push_repo_id: str | None,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    info = load_json(input_dir / "meta" / "info.json")
    old_episodes = load_jsonlines(input_dir / "meta" / "episodes.jsonl")
    old_episodes.sort(key=lambda x: x["episode_index"])

    chunks_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)
    data_path_template = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    video_path_template = info.get("video_path")
    fps = info["fps"]
    features = info["features"]

    video_keys = [key for key, ft in features.items() if ft["dtype"] == "video"]

    expected_episodes = len(old_episodes) * len(TASKS)

    # If output already has all expected episodes, just push and exit
    if output_dir.exists() and video_path_template and video_keys:
        existing_count = 0
        for ep_idx in range(expected_episodes):
            all_vids_exist = all(
                (output_dir / get_video_file_path(video_path_template, ep_idx, vk, chunks_size)).exists()
                for vk in video_keys
            )
            if all_vids_exist:
                existing_count += 1
            else:
                break

        if existing_count >= 40:
            logger.info(
                "Output directory already contains all %d episode videos — skipping processing.",
                expected_episodes,
            )
            if push_repo_id:
                push_to_hub(output_dir, push_repo_id)
            else:
                logger.info("No --push-repo-id provided; nothing to do.")
            return

        logger.info(
            "Output directory exists with %d/%d episodes — will reprocess from scratch.",
            existing_count, expected_episodes,
        )
        import ipdb; ipdb.set_trace()
        # shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI()

    logger.info("Loading LeRobotDataset '%s' from '%s' ...", repo_id, input_dir)
    dataset = LeRobotDataset(repo_id=repo_id, root=input_dir)

    if camera_keys_override:
        camera_keys = camera_keys_override
    else:
        camera_keys = dataset.meta.camera_keys
    if not camera_keys:
        raise RuntimeError("No camera features found in the dataset.")
    logger.info("Using camera keys: %s", camera_keys)

    task_entries = [{"task_index": i, "task": t} for i, t in enumerate(TASKS)]

    new_episodes = []
    new_episodes_stats = []
    global_frame_index = 0
    new_ep_index = 0
    total_new_frames = 0
    skipped_episodes = []

    for old_ep in old_episodes:
        old_ep_idx = old_ep["episode_index"]
        logger.info("=== Episode %d ===", old_ep_idx)

        parquet_rel = get_data_file_path(data_path_template, old_ep_idx, chunks_size)
        parquet_path = input_dir / parquet_rel
        if not parquet_path.exists():
            logger.warning("Parquet not found for episode %d — skipping.", old_ep_idx)
            continue

        table = pq.read_table(parquet_path)
        total_length = table.num_rows

        segments = detect_task_boundaries_vlm(
            dataset, old_ep_idx, total_length, fps,
            camera_keys, sample_every, model, client,
        )

        if segments is None:
            skipped_episodes.append(old_ep_idx)
            logger.warning("Skipping episode %d — VLM could not identify boundaries.", old_ep_idx)
            continue

        # ── Review phase: re-predict task label on each split ─────────
        logger.info("  Review phase: verifying task labels for %d segments ...", len(segments))
        for seg in segments:
            reviewed = review_segment_label(
                dataset, old_ep_idx,
                seg["start_frame"], seg["end_frame"],
                camera_keys, client, model,
            )
            if reviewed and reviewed != seg["task"]:
                logger.info(
                    "    Segment frames %d-%d: relabelled '%s' -> '%s'",
                    seg["start_frame"], seg["end_frame"], seg["task"], reviewed,
                )
                seg["task"] = reviewed
            elif reviewed:
                logger.info(
                    "    Segment frames %d-%d: confirmed '%s'",
                    seg["start_frame"], seg["end_frame"], seg["task"],
                )

        for seg in segments:
            task_name = seg["task"]
            start = seg["start_frame"]
            end = seg["end_frame"]  # inclusive
            sub_length = end - start + 1

            if sub_length <= 0:
                logger.warning("Segment '%s' has %d frames — skipping.", task_name, sub_length)
                continue

            task_idx = TASK_TO_INDEX[task_name]

            sub_table = table.slice(start, sub_length)

            col_map = {
                "episode_index": pa.array([new_ep_index] * sub_length, type=pa.int64()),
                "frame_index": pa.array(list(range(sub_length)), type=pa.int64()),
                "index": pa.array(
                    list(range(global_frame_index, global_frame_index + sub_length)),
                    type=pa.int64(),
                ),
                "task_index": pa.array([task_idx] * sub_length, type=pa.int64()),
                "timestamp": pa.array(
                    [f / fps for f in range(sub_length)], type=pa.float32()
                ),
            }

            columns = []
            for i_col, col_name in enumerate(sub_table.column_names):
                columns.append(col_map.get(col_name, sub_table.column(i_col)))

            new_sub_table = pa.table(dict(zip(sub_table.column_names, columns)))

            out_parquet_rel = get_data_file_path(data_path_template, new_ep_index, chunks_size)
            out_parquet_path = output_dir / out_parquet_rel
            out_parquet_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(new_sub_table, out_parquet_path)

            if video_path_template and video_keys:
                for vk in video_keys:
                    old_vp = input_dir / get_video_file_path(
                        video_path_template, old_ep_idx, vk, chunks_size
                    )
                    new_vp = output_dir / get_video_file_path(
                        video_path_template, new_ep_index, vk, chunks_size
                    )
                    if old_vp.exists():
                        split_video(old_vp, new_vp, start, sub_length, fps)
                    else:
                        logger.warning("Video not found: %s", old_vp)

            ep_stats = compute_sub_episode_stats(new_sub_table, features)

            new_episodes.append({
                "episode_index": new_ep_index,
                "tasks": [task_name],
                "length": sub_length,
            })
            new_episodes_stats.append({
                "episode_index": new_ep_index,
                "stats": ep_stats,
            })

            logger.info(
                "  -> new ep %d: '%s'  frames %d-%d (%d frames)",
                new_ep_index, task_name, start, end, sub_length,
            )

            global_frame_index += sub_length
            total_new_frames += sub_length
            new_ep_index += 1

    # ── Write metadata ────────────────────────────────────────────────────────
    new_info = deepcopy(info)
    new_info["total_episodes"] = new_ep_index
    new_info["total_frames"] = total_new_frames
    new_info["total_tasks"] = len(TASKS)
    new_info["total_chunks"] = (
        get_episode_chunk(new_ep_index - 1, chunks_size) + 1 if new_ep_index > 0 else 0
    )
    new_info["total_videos"] = new_ep_index * len(video_keys)
    new_info["splits"] = {"train": f"0:{new_ep_index}"}

    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    write_json(new_info, output_dir / "meta" / "info.json")
    write_jsonlines(task_entries, output_dir / "meta" / "tasks.jsonl")
    write_jsonlines(new_episodes, output_dir / "meta" / "episodes.jsonl")
    write_jsonlines(new_episodes_stats, output_dir / "meta" / "episodes_stats.jsonl")

    logger.info(
        "Done. %d original episodes -> %d new episodes (%d total frames). "
        "%d episodes skipped.",
        len(old_episodes), new_ep_index, total_new_frames, len(skipped_episodes),
    )
    if skipped_episodes:
        logger.warning("Skipped episode indices: %s", skipped_episodes)

    if push_repo_id:
        push_to_hub(output_dir, push_repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split LeRobot episodes into task sub-episodes using two-pass VLM detection."
    )
    parser.add_argument("--repo-id", type=str, required=True, help="HF repo id of the source dataset.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Source LeRobot dataset path on disk.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dataset path.")
    parser.add_argument(
        "--sample-every", type=int, default=10,
        help="Sample every N-th frame for Pass 1 classification (default: 10).",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="OpenAI vision model to use (default: gpt-4o).",
    )
    parser.add_argument(
        "--camera-key", type=str, default=None,
        help="Comma-separated camera keys to use (default: all cameras in dataset).",
    )
    parser.add_argument(
        "--push-repo-id", type=str, default=None, #"ishika/aloha_play_dataset_part_3_with_fk_full_split",
        help="If set, push the new dataset to this HF Hub repo id.",
    )
    args = parser.parse_args()

    camera_keys_override = None
    if args.camera_key:
        camera_keys_override = [k.strip() for k in args.camera_key.split(",")]

    main(
        args.repo_id, args.input_dir, args.output_dir,
        args.sample_every, args.model,
        camera_keys_override, args.push_repo_id,
    )
