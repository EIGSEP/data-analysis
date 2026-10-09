"""Render full-frame and detail views of all transmitter pixel labels."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eigsep_terrain.imageio import load_image


def render(output="transmitter_pick_audit.png",meta_file="meta.json",radius=260):
    meta=json.loads(Path(meta_file).read_text())
    keys=sorted((k for k,v in meta.items() if "transmitter_px" in v),key=int)
    fig,axes=plt.subplots(len(keys),2,figsize=(12,4*len(keys)))
    for row,key in enumerate(keys):
        image=np.flipud(load_image(f"marjum-2026-07/IMG_{key}.HEIC"))
        x,y=meta[key]["transmitter_px"];h,w=image.shape[:2]
        axes[row,0].imshow(image,origin="lower");axes[row,0].plot(x,y,"c+",ms=18,mew=2)
        axes[row,0].set_title(f"IMG_{key} full frame")
        x0=max(0,int(x-radius));x1=min(w,int(x+radius));y0=max(0,int(y-radius));y1=min(h,int(y+radius))
        axes[row,1].imshow(image[y0:y1,x0:x1],origin="lower",extent=(x0,x1,y0,y1))
        axes[row,1].plot(x,y,"c+",ms=20,mew=2);axes[row,1].set_title(f"transmitter_px=({x:.1f}, {y:.1f})")
        for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
        del image
    fig.tight_layout();fig.savefig(output,dpi=130);plt.close(fig)
    return output


if __name__ == "__main__":render()
