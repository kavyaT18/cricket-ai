

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")                  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np



PITCH_LENGTH_M = 20.12     
PITCH_WIDTH_M  = 3.05     
CREASE_TO_STUMP = 1.22    

# Length zone boundaries (distance from batting crease, metres)
LENGTH_ZONES = {
    "Yorker":       (0.0,  1.2,  "#e74c3c"),   # red
    "Full":         (1.2,  3.5,  "#e67e22"),   # orange
    "Good length":  (3.5,  6.0,  "#2ecc71"),   # green
    "Short":        (6.0,  8.5,  "#3498db"),   # blue
    "Long hop":     (8.5,  20.12,"#9b59b6"),   # purple
}

# Line zone boundaries (offset from centre of stumps, metres, +ve = off side)
LINE_ZONES = {
    "Wide off":    (0.9,   3.0,  "#f39c12"),
    "Off stump":   (0.0,   0.9,  "#27ae60"),
    "Middle":     (-0.6,   0.0,  "#2980b9"),
    "Leg stump":  (-1.2,  -0.6,  "#8e44ad"),
    "Wide leg":   (-3.0,  -1.2,  "#c0392b"),
}

# Canvas pixel dimensions for the pitch diagram
CANVAS_W = 340   
CANVAS_H = 680   



@dataclass
class Delivery:
   
    length_m: float
    line_m:   float
    label:    str = ""


@dataclass
class PitchMapResult:
    """Everything produced by PitchMap.render()."""
    fig:            plt.Figure | None = None
    length_summary: dict[str, float] = field(default_factory=dict)   # zone → %
    line_summary:   dict[str, float] = field(default_factory=dict)
    total_deliveries: int = 0




def metres_to_pixel(length_m: float, line_m: float,
                    canvas_h: int = CANVAS_H,
                    canvas_w: int = CANVAS_W) -> tuple[float, float]:
    
    length_m = np.clip(length_m, 0.0, PITCH_LENGTH_M)
    line_m   = np.clip(line_m,  -PITCH_WIDTH_M / 2, PITCH_WIDTH_M / 2)

    # Margins (pixels) around the pitch rectangle
    margin_y = canvas_h * 0.06
    margin_x = canvas_w * 0.15

    pitch_px_h = canvas_h - 2 * margin_y
    pitch_px_w = canvas_w - 2 * margin_x

    px_y = canvas_h - margin_y - (length_m / PITCH_LENGTH_M) * pitch_px_h
    px_x = margin_x + (line_m + PITCH_WIDTH_M / 2) / PITCH_WIDTH_M * pitch_px_w

    return px_x, px_y


def pixel_to_metres(px_x: float, px_y: float,
                    canvas_h: int = CANVAS_H,
                    canvas_w: int = CANVAS_W) -> tuple[float, float]:
    """Inverse of metres_to_pixel — useful for mapping detected pixel coords."""
    margin_y = canvas_h * 0.06
    margin_x = canvas_w * 0.15

    pitch_px_h = canvas_h - 2 * margin_y
    pitch_px_w = canvas_w - 2 * margin_x

    length_m = (canvas_h - margin_y - px_y) / pitch_px_h * PITCH_LENGTH_M
    line_m   = (px_x - margin_x) / pitch_px_w * PITCH_WIDTH_M - PITCH_WIDTH_M / 2

    return float(np.clip(length_m, 0, PITCH_LENGTH_M)), float(line_m)



class PitchMap:
   

    def __init__(self):
        self.deliveries: list[Delivery] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def add_delivery(self, length_m: float, line_m: float, label: str = ""):
        """Add one ball's bounce point."""
        self.deliveries.append(Delivery(length_m=length_m, line_m=line_m, label=label))

    def add_deliveries(self, deliveries: Sequence[tuple[float, float]]):
        """Bulk add list of (length_m, line_m) tuples."""
        for d in deliveries:
            self.add_delivery(d[0], d[1])

    def from_pixel_bounce(self, px: int, py: int,
                          frame_w: int, frame_h: int, label: str = ""):
      
        length_m = (1.0 - py / frame_h) * PITCH_LENGTH_M
        line_m   = (px / frame_w - 0.5) * PITCH_WIDTH_M
        self.add_delivery(length_m, line_m, label)

    def clear(self):
        self.deliveries = []

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(
        self,
        save_path: str | None = None,
        title:     str        = "Pitch Map",
        show_heatmap:  bool   = True,
        show_dots:     bool   = True,
        show_zones:    bool   = True,
        dpi:           int    = 130,
    ) -> PitchMapResult:
       
        result = PitchMapResult()
        result.total_deliveries = len(self.deliveries)

        fig, ax = plt.subplots(figsize=(4.5, 9), dpi=dpi)
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        self._draw_pitch(ax)

        if show_zones:
            self._draw_length_zones(ax)

        if self.deliveries:
            if show_heatmap:
                self._draw_heatmap(ax)
            if show_dots:
                self._draw_dots(ax)
            result.length_summary = self._length_summary()
            result.line_summary   = self._line_summary()
            self._draw_summary_table(ax, result.length_summary)

        ax.set_xlim(0, CANVAS_W)
        ax.set_ylim(0, CANVAS_H)
        ax.set_aspect("equal")
        ax.axis("off")

        ax.set_title(
            f"{title}  ({len(self.deliveries)} deliveries)",
            color="white", fontsize=11, fontweight="bold", pad=10
        )

        fig.tight_layout(pad=0.5)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"[pitch_map] Saved → {save_path}")

        result.fig = fig
        return result

    

    def _draw_pitch(self, ax: plt.Axes):
        """Draw the pitch rectangle, creases, stumps, and labels."""
        # Pitch boundary corners in pixel space
        tl = metres_to_pixel(PITCH_LENGTH_M, -PITCH_WIDTH_M / 2)
        br = metres_to_pixel(0.0,             PITCH_WIDTH_M / 2)

        pitch_rect = mpatches.FancyBboxPatch(
            (tl[0], tl[1]),
            br[0] - tl[0], br[1] - tl[1],
            boxstyle="round,pad=2",
            linewidth=1.5, edgecolor="#aaaaaa",
            facecolor="#2d5a27",           # grass green
            zorder=1,
        )
        ax.add_patch(pitch_rect)

       
        bc_l = metres_to_pixel(0.0, -PITCH_WIDTH_M / 2)
        bc_r = metres_to_pixel(0.0,  PITCH_WIDTH_M / 2)
        ax.plot([bc_l[0], bc_r[0]], [bc_l[1], bc_r[1]],
                color="white", lw=1.5, zorder=4)

       
        bwl_l = metres_to_pixel(PITCH_LENGTH_M, -PITCH_WIDTH_M / 2)
        bwl_r = metres_to_pixel(PITCH_LENGTH_M,  PITCH_WIDTH_M / 2)
        ax.plot([bwl_l[0], bwl_r[0]], [bwl_l[1], bwl_r[1]],
                color="white", lw=1.5, zorder=4)

        # Popping crease (batting end, 1.22 m up from batting crease)
        pc_l = metres_to_pixel(CREASE_TO_STUMP, -PITCH_WIDTH_M / 2)
        pc_r = metres_to_pixel(CREASE_TO_STUMP,  PITCH_WIDTH_M / 2)
        ax.plot([pc_l[0], pc_r[0]], [pc_l[1], pc_r[1]],
                color="white", lw=1.0, ls="--", alpha=0.5, zorder=4)

        # ── Stumps ────────────────────────────────────────────────────────────
        self._draw_stumps(ax, end="batting")
        self._draw_stumps(ax, end="bowling")

        # ── Labels ────────────────────────────────────────────────────────────
        bc_mid = metres_to_pixel(0.0, 0.0)
        ax.text(bc_mid[0], bc_mid[1] + 22, "BATSMAN",
                ha="center", va="center", color="white",
                fontsize=7, alpha=0.7, zorder=5)

        bwl_mid = metres_to_pixel(PITCH_LENGTH_M, 0.0)
        ax.text(bwl_mid[0], bwl_mid[1] - 22, "BOWLER",
                ha="center", va="center", color="white",
                fontsize=7, alpha=0.7, zorder=5)

    def _draw_stumps(self, ax: plt.Axes, end: str):
        """Draw three stump dots at each end."""
        y_m = CREASE_TO_STUMP if end == "batting" else PITCH_LENGTH_M - CREASE_TO_STUMP
        offsets = [-0.11, 0.0, 0.11]    # three stumps, 11 cm apart
        for off in offsets:
            px, py = metres_to_pixel(y_m, off)
            ax.plot(px, py, "s", color="#f0c040", ms=4, zorder=6)

    def _draw_length_zones(self, ax: plt.Axes):
        """Shade each length zone with a subtle colour band."""
        for zone_name, (lo, hi, colour) in LENGTH_ZONES.items():
            lo_l = metres_to_pixel(lo, -PITCH_WIDTH_M / 2)
            hi_r = metres_to_pixel(hi,  PITCH_WIDTH_M / 2)

            rect = mpatches.Rectangle(
                (lo_l[0], hi_r[1]),              # note y-axis is flipped
                hi_r[0] - lo_l[0],
                lo_l[1] - hi_r[1],
                linewidth=0, facecolor=colour,
                alpha=0.12, zorder=2,
            )
            ax.add_patch(rect)

            # Zone label on the left margin
            mid_y = (lo_l[1] + hi_r[1]) / 2
            ax.text(lo_l[0] - 6, mid_y, zone_name,
                    ha="right", va="center", color=colour,
                    fontsize=6.5, alpha=0.9, zorder=5)

    def _draw_heatmap(self, ax: plt.Axes):
        """KDE heatmap over ball landing positions."""
        if len(self.deliveries) < 3:
            return   # not enough data for KDE

        # Convert to pixel coordinates
        xs = [metres_to_pixel(d.length_m, d.line_m)[0] for d in self.deliveries]
        ys = [metres_to_pixel(d.length_m, d.line_m)[1] for d in self.deliveries]

        # Build a 2D histogram, smooth with Gaussian
        from scipy.ndimage import gaussian_filter
        bins_x, bins_y = 40, 80
        h, xedges, yedges = np.histogram2d(xs, ys, bins=[bins_x, bins_y],
                                            range=[[0, CANVAS_W], [0, CANVAS_H]])
        h_smooth = gaussian_filter(h.T, sigma=2.5)

        # Custom colormap: transparent → hot
        cmap = LinearSegmentedColormap.from_list(
            "cricket_heat",
            [(0, (0, 0, 0, 0)),
             (0.3, (0.2, 0.8, 0.2, 0.3)),
             (0.6, (1.0, 0.8, 0.0, 0.55)),
             (1.0, (1.0, 0.1, 0.1, 0.75))],
        )

        ax.imshow(
            h_smooth,
            extent=[0, CANVAS_W, 0, CANVAS_H],
            origin="lower",
            cmap=cmap,
            aspect="auto",
            zorder=3,
            interpolation="bilinear",
        )

    def _draw_dots(self, ax: plt.Axes):
        """Plot each delivery as a colour-coded dot."""
        for d in self.deliveries:
            colour = _length_colour(d.length_m)
            px, py = metres_to_pixel(d.length_m, d.line_m)
            ax.plot(px, py, "o",
                    color=colour, ms=5,
                    markeredgecolor="white", markeredgewidth=0.4,
                    alpha=0.85, zorder=7)

    def _draw_summary_table(self, ax: plt.Axes, length_summary: dict[str, float]):
        """Small legend/summary in the bottom margin."""
        x_start = 18
        y_start = CANVAS_H * 0.04
        ax.text(CANVAS_W / 2, y_start + 10, "Length breakdown",
                ha="center", va="center", color="white",
                fontsize=7.5, fontweight="bold", zorder=8)

        for i, (zone, pct) in enumerate(length_summary.items()):
            colour = LENGTH_ZONES[zone][2]
            x = x_start + i * (CANVAS_W - x_start * 2) / len(length_summary)
            ax.plot(x, y_start - 4, "o", color=colour, ms=6, zorder=8)
            ax.text(x + 8, y_start - 4, f"{zone}\n{pct:.0f}%",
                    va="center", color=colour, fontsize=5.5, zorder=8)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _length_summary(self) -> dict[str, float]:
        """Return % of deliveries in each length zone."""
        counts = {z: 0 for z in LENGTH_ZONES}
        for d in self.deliveries:
            for zone, (lo, hi, _) in LENGTH_ZONES.items():
                if lo <= d.length_m < hi:
                    counts[zone] += 1
                    break
        total = len(self.deliveries) or 1
        return {z: round(c / total * 100, 1) for z, c in counts.items()}

    def _line_summary(self) -> dict[str, float]:
        """Return % of deliveries in each line zone."""
        counts = {z: 0 for z in LINE_ZONES}
        for d in self.deliveries:
            for zone, (lo, hi, _) in LINE_ZONES.items():
                if lo <= d.line_m < hi:
                    counts[zone] += 1
                    break
        total = len(self.deliveries) or 1
        return {z: round(c / total * 100, 1) for z, c in counts.items()}

    def print_summary(self):
        """Print a text summary to stdout."""
        if not self.deliveries:
            print("[pitch_map] No deliveries recorded.")
            return
        ls = self._length_summary()
        lns = self._line_summary()
        print(f"\n{'='*40}")
        print(f"  Pitch Map Summary  ({len(self.deliveries)} deliveries)")
        print(f"{'='*40}")
        print("\nLength zones:")
        for zone, pct in ls.items():
            bar = "█" * int(pct / 5)
            colour_label = zone
            print(f"  {colour_label:<14} {pct:5.1f}%  {bar}")
        print("\nLine zones:")
        for zone, pct in lns.items():
            bar = "█" * int(pct / 5)
            print(f"  {zone:<14} {pct:5.1f}%  {bar}")
        print()




def build_from_ball_tracker(tracker_result, shot_label: str = "") -> PitchMap:
   
    pm = PitchMap()

    if tracker_result.bounce_point is None:
        return pm

   
    bx, by = tracker_result.bounce_point

   
    frame_h = 720
    frame_w = 1280

    pm.from_pixel_bounce(bx, by, frame_w, frame_h, label=shot_label)
    return pm


def build_demo_map(n: int = 30, seed: int = 42) -> PitchMap:
  
    rng = np.random.default_rng(seed)
    pm  = PitchMap()

    # Good-length cluster (majority of deliveries)
    n_good = int(n * 0.45)
    for _ in range(n_good):
        length = rng.normal(loc=5.0, scale=0.6)
        line   = rng.normal(loc=0.3, scale=0.25)   # off stump line
        pm.add_delivery(np.clip(length, 0, 20), line, "good_length")

    # Full deliveries
    n_full = int(n * 0.25)
    for _ in range(n_full):
        length = rng.normal(loc=2.5, scale=0.5)
        line   = rng.normal(loc=0.2, scale=0.3)
        pm.add_delivery(np.clip(length, 0, 20), line, "full")

    # Short balls
    n_short = int(n * 0.20)
    for _ in range(n_short):
        length = rng.normal(loc=7.2, scale=0.7)
        line   = rng.normal(loc=0.0, scale=0.4)
        pm.add_delivery(np.clip(length, 0, 20), line, "short")

    # Yorkers
    n_york = n - n_good - n_full - n_short
    for _ in range(n_york):
        length = rng.normal(loc=0.7, scale=0.3)
        line   = rng.normal(loc=0.1, scale=0.2)
        pm.add_delivery(np.clip(length, 0, 20), line, "yorker")

    return pm



def _length_colour(length_m: float) -> str:
    for _, (lo, hi, colour) in LENGTH_ZONES.items():
        if lo <= length_m < hi:
            return colour
    return "#ffffff"



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cricket pitch map generator")
    parser.add_argument("--demo",   action="store_true",
                        help="Generate a demo pitch map with synthetic data")
    parser.add_argument("--n",      type=int,   default=40,
                        help="Number of synthetic deliveries (default 40)")
    parser.add_argument("--output", type=str,   default="output/pitch_map.png",
                        help="Output PNG path")
    parser.add_argument("--title",  type=str,   default="Pitch Map")
    args = parser.parse_args()

    if args.demo:
        print(f"[demo] Generating pitch map with {args.n} synthetic deliveries...")
        pm = build_demo_map(n=args.n)
        pm.print_summary()
        result = pm.render(save_path=args.output, title=args.title)
        print(f"[demo] Saved to {args.output}")
    else:
        parser.print_help()
        print("\nExample:")
        print("  python src/pitch_map.py --demo --n 50 --output output/pitch_map.png")