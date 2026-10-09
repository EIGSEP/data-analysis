"""Interactive, distortion-aware landmark picking for the Marjum images."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from marjum_camera import project


class LandmarkPicker:
    """Pick multiple landmark types without loading every image at once.

    ``landmarks`` maps each display name to a dictionary containing
    ``pixel_field`` and, optionally, ``fit_file``, ``position_field``, and
    ``color``. A missing physical fit intentionally gives a full-frame view.
    """

    def __init__(self, image_dir="marjum-2026-07",
                 camera_fit="cv_antenna_repick_v1/fit_antenna.npz",
                 landmarks=None, meta_file="meta.json", radius_m=3.0):
        self.image_dir = Path(image_dir)
        self.camera_fit = Path(camera_fit)
        self.meta_file = Path(meta_file)
        self.radius_m = float(radius_m)
        self.landmarks = landmarks or {
            "antenna": {
                "pixel_field": "ant_px", "fit_file": str(self.camera_fit),
                "position_field": "antenna", "color": "magenta",
            }
        }
        self.images = {
            p.stem.removeprefix("IMG_"): p
            for p in sorted(self.image_dir.glob("IMG_*.HEIC"))
        }
        if not self.images:
            raise FileNotFoundError(f"No IMG_*.HEIC files found in {self.image_dir}")
        self.keys = sorted(self.images, key=lambda value: int(value))
        self.poses, self.shapes, self.distortion = self._load_camera_fit()
        self.positions = {
            name: self._load_landmark_position(spec)
            for name, spec in self.landmarks.items()
        }
        self.meta = self._load_meta()
        self._figure = None
        self._marker = None
        self._connection = None

    def _load_camera_fit(self):
        if not self.camera_fit.exists():
            return {}, {}, {}
        with np.load(self.camera_fit, allow_pickle=False) as fit:
            required = {"keys", "cameras", "shapes"}
            missing = required.difference(fit.files)
            if missing:
                raise KeyError(f"{self.camera_fit} lacks {sorted(missing)}")
            keys = [str(key) for key in fit["keys"]]
            cameras = np.asarray(fit["cameras"], float)
            shapes = np.asarray(fit["shapes"], int)
            distortion = (np.asarray(fit["distortion"], float)
                          if "distortion" in fit.files
                          else np.zeros((len(keys), 2), float))
        if not (len(keys) == len(cameras) == len(shapes) == len(distortion)):
            raise ValueError(f"Inconsistent camera arrays in {self.camera_fit}")
        return (dict(zip(keys, cameras)), dict(zip(keys, shapes)),
                dict(zip(keys, distortion)))

    @staticmethod
    def _load_landmark_position(spec):
        direct = spec.get("position")
        if direct is not None:
            return np.asarray(direct, float)
        filename = spec.get("fit_file")
        field = spec.get("position_field")
        if not filename or not field or not Path(filename).exists():
            return None
        with np.load(filename, allow_pickle=False) as fit:
            if field not in fit.files:
                return None
            position = np.asarray(fit[field], float)
        if position.shape == (3,) and np.all(np.isfinite(position)):
            return position
        return None

    def _load_meta(self):
        if not self.meta_file.exists():
            return {}
        with self.meta_file.open() as stream:
            return json.load(stream)

    def set_pixel(self, key, target, xy):
        """Save one label immediately while preserving other landmark fields."""
        field = self.landmarks[target]["pixel_field"]
        latest = self._load_meta()
        latest.setdefault(str(key), {})[field] = [float(xy[0]), float(xy[1])]
        temporary = self.meta_file.with_name(self.meta_file.name + ".tmp")
        with temporary.open("w") as stream:
            json.dump(latest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, self.meta_file)
        self.meta = latest
        return latest[str(key)][field]

    @staticmethod
    def _transverse_ring(camera_xyz, landmark_xyz, radius_m, count=48):
        sightline = np.asarray(landmark_xyz, float) - np.asarray(camera_xyz, float)
        distance = np.linalg.norm(sightline)
        if not np.isfinite(distance) or distance <= radius_m:
            return None, distance
        sightline /= distance
        reference = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(sightline, reference)) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        axis1 = np.cross(sightline, reference)
        axis1 /= np.linalg.norm(axis1)
        axis2 = np.cross(sightline, axis1)
        angle = np.linspace(0, 2 * np.pi, count, endpoint=False)
        ring = landmark_xyz + radius_m * (
            np.cos(angle)[:, None] * axis1 + np.sin(angle)[:, None] * axis2)
        return ring, distance

    def expected_crop(self, key, target, actual_shape=None):
        """Return ``(bounds, center, reason)`` for a physical-radius crop.

        Bounds are integer ``(x0, x1, y0, y1)`` in bottom-up native pixels.
        ``bounds`` is None whenever the safe behavior is full-frame.
        """
        key = str(key)
        exclude = self.landmarks[target].get("exclude_keys")
        if exclude and key in exclude:
            return None, None, f"{key} is known not to see this landmark"
        position = self.positions.get(target)
        if position is None:
            return None, None, "no current physical solution"
        if key not in self.poses:
            return None, None, "image has no fitted pose"
        fit_shape = np.asarray(self.shapes[key], int)
        if actual_shape is not None and tuple(fit_shape) != tuple(actual_shape):
            return (None, None,
                    f"fit shape {tuple(fit_shape)} != image shape {tuple(actual_shape)}")
        pose = self.poses[key]
        radial = self.distortion.get(key, np.zeros(2))
        center, depth = project(pose, fit_shape, position, radial)
        center = center[0]
        h, w = fit_shape
        if depth[0] <= 0 or not np.all(np.isfinite(center)):
            return None, None, "physical solution is behind this camera"
        if not (0 <= center[0] < w and 0 <= center[1] < h):
            return None, center, "physical solution projects outside the image"
        ring, distance = self._transverse_ring(pose[:3], position, self.radius_m)
        if ring is None:
            return None, center, "camera is too close for a transverse crop"
        edge, edge_depth = project(pose, fit_shape, ring, radial)
        valid = np.all(np.isfinite(edge), axis=1) & (edge_depth > 0)
        if valid.sum() < 8:
            return None, center, "physical crop does not project safely"
        edge = edge[valid]
        # 8% visual context; the floor matters only for very distant targets.
        half_x = max(float(np.max(np.abs(edge[:, 0] - center[0]))), 32.0)
        half_y = max(float(np.max(np.abs(edge[:, 1] - center[1]))), 32.0)
        x0 = max(0, int(np.floor(center[0] - 1.08 * half_x)))
        x1 = min(w, int(np.ceil(center[0] + 1.08 * half_x)) + 1)
        y0 = max(0, int(np.floor(center[1] - 1.08 * half_y)))
        y1 = min(h, int(np.ceil(center[1] + 1.08 * half_y)) + 1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None, center, "projected crop is degenerate"
        reason = f"±{self.radius_m:g} m crop; fitted range {distance:.1f} m"
        return (x0, x1, y0, y1), center, reason

    def summary(self):
        fitted = sum(key in self.poses for key in self.keys)
        lines = [f"{len(self.keys)} images; {fitted} have poses in {self.camera_fit}",
                 f"metadata: {self.meta_file}"]
        for name, spec in self.landmarks.items():
            count = sum(spec["pixel_field"] in self.meta.get(key, {})
                        for key in self.keys)
            state = ("physical solution loaded" if self.positions[name] is not None
                     else "no physical solution (full frame)")
            lines.append(f"{name}: {count} picks in {spec['pixel_field']!r}; {state}")
        return "\n".join(lines)

    def widget(self):
        """Build and display the one-image-at-a-time Jupyter picker."""
        import ipywidgets as widgets
        from IPython.display import clear_output, display
        from eigsep_terrain.imageio import load_image

        target = widgets.Dropdown(options=list(self.landmarks), description="Landmark:")
        image = widgets.Dropdown(options=self.keys, description="Image:")
        full = widgets.Checkbox(value=False, description="Full frame")
        previous = widgets.Button(description="Previous", icon="arrow-left")
        following = widgets.Button(description="Next", icon="arrow-right")
        status = widgets.HTML()
        output = widgets.Output()

        def move(step):
            index = self.keys.index(image.value)
            image.value = self.keys[(index + step) % len(self.keys)]

        previous.on_click(lambda _: move(-1))
        following.on_click(lambda _: move(+1))

        def render(*_):
            if self._figure is not None:
                plt.close(self._figure)
            key, name = image.value, target.value
            rgb = np.flipud(load_image(self.images[key]))
            bounds, expected, reason = self.expected_crop(key, name, rgb.shape[:2])
            use_crop = bounds is not None and not full.value
            if use_crop:
                x0, x1, y0, y1 = bounds
                shown = rgb[y0:y1, x0:x1]
                extent = (x0, x1, y0, y1)
                view = reason
            else:
                h, w = rgb.shape[:2]
                shown = rgb
                extent = (0, w, 0, h)
                view = "full frame (requested)" if full.value else f"full frame: {reason}"
            with output:
                clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(10, 8))
                self._figure = fig
                ax.imshow(shown, origin="lower", extent=extent)
                color = self.landmarks[name].get("color", "magenta")
                field = self.landmarks[name]["pixel_field"]
                existing = self.meta.get(key, {}).get(field)
                self._marker = None
                if expected is not None:
                    ax.plot(*expected, "x", color="yellow", ms=13, mew=2,
                            label="fit prediction")
                if existing is not None:
                    self._marker = ax.plot(*existing, "+", color=color, ms=18,
                                           mew=2.5, label=field)[0]
                if expected is not None or existing is not None:
                    ax.legend(loc="best")
                ax.set_xlabel("native x pixel")
                ax.set_ylabel("native y pixel (bottom-up)")
                ax.set_title(f"IMG_{key}: click the {name} — {view}")

                def onclick(event):
                    if (event.inaxes is not ax or event.xdata is None
                            or event.ydata is None):
                        return
                    saved = self.set_pixel(key, name, (event.xdata, event.ydata))
                    if self._marker is not None:
                        self._marker.remove()
                    self._marker = ax.plot(*saved, "+", color=color, ms=18,
                                           mew=2.5, label=field)[0]
                    ax.set_title(
                        f"IMG_{key}: {field}=({saved[0]:.1f}, {saved[1]:.1f}) "
                        f"saved to {self.meta_file}")
                    status.value = (f"<b>Saved:</b> IMG_{key} {field} = "
                                    f"({saved[0]:.1f}, {saved[1]:.1f})")
                    fig.canvas.draw_idle()

                self._connection = fig.canvas.mpl_connect("button_press_event", onclick)
                plt.show()
            del rgb, shown
            picked = ("not yet picked" if existing is None else
                      f"current {field}=({existing[0]:.1f}, {existing[1]:.1f})")
            status.value = f"<b>IMG_{key}</b> — {view}; {picked}"

        target.observe(render, names="value")
        image.observe(render, names="value")
        full.observe(render, names="value")
        ui = widgets.VBox([
            widgets.HBox([target, image, previous, following, full]), status, output])
        display(ui)
        render()
        return ui
