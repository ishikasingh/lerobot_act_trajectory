#!/usr/bin/env python3
"""
Molmo EE-prediction server.

Runs on a GPU machine and serves Molmo predictions over HTTP.
The robot-side client (control_robot_remote.py) sends camera images
and proprio state, and receives predicted EE trajectories back.

Usage:
    python lerobot/scripts/molmo_server.py \
        --checkpoint ~/molmo_trajectory/molmo/finetuned_model/ \
        --host 0.0.0.0 --port 5050

The server exposes a single POST endpoint:

    POST /predict
        Body (JSON):
            image_b64:      base64-encoded JPEG of the camera frame
            instruction:    language instruction string
            proprio_state:  list of floats (optional, e.g. 6-d EE positions)
        Response (JSON):
            left_ee:   list[list[float]]  (action_horizon x 3)
            right_ee:  list[list[float]]  (action_horizon x 3)
"""

import argparse
import base64
import io
import logging
import time

import numpy as np
import torch
from flask import Flask, request, jsonify
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
app = Flask(__name__)

_predictor = None  # Will be set in main()


class MolmoPredictor:
    """Wraps Molmo model loading and inference (identical logic to MolmoEEPredictor)."""

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
        logging.info(
            f"Model loaded: action_horizon={self.action_horizon}, action_dim={self.action_dim}"
        )

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        proprio_state: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        pil_image = Image.fromarray(image)
        example = {
            "image": pil_image,
            "prompt": instruction,
            "style": "trajectory_3d_egodex_trossen_direct",
        }
        if proprio_state is not None:
            example["state"] = proprio_state
            example["proprio_state"] = proprio_state

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
        left_ee = actions_np[:, :3].astype(np.float32)
        right_ee = actions_np[:, 3:6].astype(np.float32)
        return left_ee, right_ee


@app.route("/predict", methods=["POST"])
def handle_predict():
    t0 = time.time()
    data = request.get_json(force=True)

    # Decode JPEG image from base64
    image_bytes = base64.b64decode(data["image_b64"])
    image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

    instruction = data.get("instruction", "")
    proprio_raw = data.get("proprio_state")
    proprio_state = np.array(proprio_raw, dtype=np.float32) if proprio_raw is not None else None

    left_ee, right_ee = _predictor.predict(image, instruction, proprio_state)

    elapsed = time.time() - t0
    logging.info(f"/predict  {image.shape}  dt={elapsed:.3f}s")

    return jsonify({
        "left_ee": left_ee.tolist(),
        "right_ee": right_ee.tolist(),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def main():
    parser_cli = argparse.ArgumentParser(description="Molmo EE-prediction server")
    parser_cli.add_argument("--checkpoint", type=str, required=True, help="Path to Molmo checkpoint")
    parser_cli.add_argument("--device", type=str, default="cuda")
    parser_cli.add_argument("--num_ode_steps", type=int, default=10)
    parser_cli.add_argument("--host", type=str, default="0.0.0.0")
    parser_cli.add_argument("--port", type=int, default=5050)
    args = parser_cli.parse_args()

    global _predictor
    _predictor = MolmoPredictor(
        checkpoint=args.checkpoint,
        device=args.device,
        num_ode_steps=args.num_ode_steps,
    )

    logging.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
