#!/usr/bin/env python3
"""
Molmo EE-prediction API server.

FastAPI server with /predict and /health. Use --use-pinggy to expose via
Pinggy tunnel (public URL for cross-network access).

Usage (local/same network):
    python lerobot/scripts/molmo_server.py --checkpoint /path/to/molmo --port 5050

Usage (cross-network with Pinggy):
    python lerobot/scripts/molmo_server.py --checkpoint /path/to/molmo --port 5050 --use-pinggy
    # Prints public URL like https://xxx.a0.pinggy.io - use that for --control.molmo_checkpoint
"""

import argparse
import base64
import io
import logging
import threading
import time

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_predictor = None


class MolmoPredictor:
    """Wraps Molmo model loading and inference (identical logic to MolmoEEPredictor)."""

    def __init__(self, checkpoint: str, device: str = "cuda", num_ode_steps: int = 10):
        import torch
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

    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        proprio_state: np.ndarray | None = None,
    ) -> np.ndarray:
        import torch

        with torch.no_grad():
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
            return actions_np[:, :6].astype(np.float32)


def create_app():
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="Molmo EE-prediction")

    class PredictRequest(BaseModel):
        image_b64: str
        instruction: str = ""
        proprio_state: list[float] | None = None

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    def predict(req: PredictRequest):
        t0 = time.time()
        image_bytes = base64.b64decode(req.image_b64)
        image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        instruction = req.instruction or ""
        proprio_state = np.array(req.proprio_state, dtype=np.float32) if req.proprio_state else None

        trajectory = _predictor.predict(image, instruction, proprio_state)

        elapsed = time.time() - t0
        logging.info(f"/predict  {image.shape}  dt={elapsed:.3f}s")
        return {"trajectory": trajectory.tolist()}

    return app


def run_pinggy_tunnel(port: int):
    """Start Pinggy tunnel, print public URL. Blocks until server is up."""
    import pinggy
    tunnel = pinggy.start_tunnel(forwardto=f"localhost:{port}")
    urls = getattr(tunnel, "urls", None) or [str(tunnel)]
    for u in urls:
        logging.info(f"Pinggy tunnel: {u}")
    print(f"\n>>> Use this URL for --control.molmo_checkpoint: {urls[0]}\n")
    return tunnel


def main():
    global _predictor

    parser = argparse.ArgumentParser(description="Molmo EE-prediction API server")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Molmo checkpoint")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--use-pinggy", action="store_true", help="Expose via Pinggy tunnel (cross-network)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_ode_steps", type=int, default=10)
    args = parser.parse_args()

    _predictor = MolmoPredictor(
        checkpoint=args.checkpoint,
        device=args.device,
        num_ode_steps=args.num_ode_steps,
    )

    app = create_app()

    if args.use_pinggy:
        # Start server in background, then start Pinggy tunnel (needs server up first)
        def run_server():
            import uvicorn
            uvicorn.run(app, host=args.host, port=args.port)

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(2)  # Wait for server to bind

        try:
            run_pinggy_tunnel(args.port)  # Blocks until Ctrl+C
        except Exception as e:
            logging.error(f"Pinggy failed: {e}. Run manually: pinggy http {args.port}")
    else:
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
