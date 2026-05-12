"""Write MuJoCo camera frames to a folder for NaVILA sequential client.

Place this file at:
    src/holosoma/holosoma/simulator/mujoco/navila_camera_streamer.py

It renders a named MuJoCo camera at a low fixed rate and writes images with
monotonic filenames. The folder can be consumed by navila_folder_stream_client.py
with --ingest-mode sequential --require-full-window.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import mujoco
from loguru import logger


class NavilaCameraStreamWriter:
    """Low-rate RGB frame writer for a named MuJoCo camera.

    Parameters
    ----------
    simulator:
        Holosoma MuJoCo simulator instance. It must have root_model and backend.
    camera_name:
        Name of the camera in the compiled MuJoCo model, usually "robot_head_nav".
    out_dir:
        Directory where frames are written.
    width, height:
        Render resolution.
    interval_sec:
        Minimum wall-clock interval between saved frames.
    clean_start:
        If True, remove old frame_*.jpg files at startup.
    keep_max:
        If positive, keep only the latest keep_max frames in out_dir.
    """

    def __init__(
        self,
        simulator,
        camera_name: str = "robot_head_nav",
        out_dir: str | Path = "/tmp/navila_mujoco_stream",
        width: int = 640,
        height: int = 480,
        interval_sec: float = 0.5,
        clean_start: bool = True,
        keep_max: int = 0,
    ) -> None:
        self.simulator = simulator
        self.camera_name = camera_name
        self.out_dir = Path(out_dir).expanduser()
        self.width = int(width)
        self.height = int(height)
        self.interval_sec = float(interval_sec)
        self.keep_max = int(keep_max)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        if clean_start:
            # Wipe every jpg (and any leftover atomic-write temp file) regardless
            # of prefix, so a previous real-camera run does not leak frames into
            # this session's window.
            for pattern in ("*.jpg", ".*.jpg", ".*.tmp"):
                for p in self.out_dir.glob(pattern):
                    try:
                        p.unlink()
                    except OSError:
                        pass

        assert self.simulator.root_model is not None
        self._renderer = mujoco.Renderer(self.simulator.root_model, height=self.height, width=self.width)
        self._frame_idx = 0
        self._start_time = time.time()
        self._next_time = 0.0

        self._check_camera_exists()
        logger.info(
            "NaVILA MuJoCo camera stream enabled: camera='{}', dir='{}', size={}x{}, interval={}s",
            self.camera_name,
            self.out_dir,
            self.width,
            self.height,
            self.interval_sec,
        )

    def _check_camera_exists(self) -> None:
        model = self.simulator.root_model
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if cam_id < 0:
            names = []
            for i in range(model.ncam):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                names.append(name)
            raise RuntimeError(
                f"Camera '{self.camera_name}' not found in compiled MuJoCo model. "
                f"Available cameras: {names}"
            )

    def maybe_write(self) -> Optional[Path]:
        """Render and write one frame if interval_sec has elapsed."""
        now = time.time()
        if now < self._next_time:
            return None
        self._next_time = now + self.interval_sec

        render_data = self.simulator.backend.get_render_data(world_id=0)
        self._renderer.update_scene(render_data, camera=self.camera_name)
        frame = self._renderer.render()

        if frame is None or len(frame.shape) != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"Unexpected rendered frame shape: {None if frame is None else frame.shape}")

        self._frame_idx += 1
        elapsed_ms = int((now - self._start_time) * 1000)
        final_path = self.out_dir / f"frame_{self._frame_idx:06d}_t{elapsed_ms:010d}ms.jpg"
        tmp_path = self.out_dir / f".{final_path.name}.tmp.jpg"

        imageio.imwrite(tmp_path, frame, quality=90)
        os.replace(tmp_path, final_path)

        if self.keep_max > 0:
            # Cap by total jpgs in the directory so foreign-prefixed leftovers
            # cannot push the rolling window above keep_max.
            jpgs = sorted(p for p in self.out_dir.glob("*.jpg") if not p.name.startswith("."))
            old = jpgs[:-self.keep_max] if len(jpgs) > self.keep_max else []
            for p in old:
                try:
                    p.unlink()
                except OSError:
                    pass

        logger.info("[NAVILA_CAMERA_STREAM] wrote {}", final_path)
        return final_path
