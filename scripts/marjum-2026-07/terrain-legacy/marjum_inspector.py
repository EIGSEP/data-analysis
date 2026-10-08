"""Memory-bounded landmark comparison panels for the fit inspector."""
import numpy as np
import matplotlib.pyplot as plt
from eigsep_terrain.imageio import load_image


def plot_landmark_comparison(rows, fit_file, landmark='transmitter', ncols=4, pad=60):
    rows=sorted(rows,key=lambda r:-r[-1])
    fig,axes=plt.subplots(max(1,int(np.ceil(len(rows)/ncols))),ncols,
                          figsize=(3.4*ncols,3.0*max(1,int(np.ceil(len(rows)/ncols)))))
    axes=np.atleast_1d(axes).ravel()
    for ax,(key,x,y,px,py,resid) in zip(axes,rows):
        image=np.flipud(load_image(f'marjum-2026-07/IMG_{key}.HEIC'))
        h,w=image.shape[:2]
        if not np.isfinite([x,y,px,py]).all():
            ax.set_title(f'{key}: invalid projection');ax.axis('off');continue
        x0=int(np.clip(min(x,px)-pad,0,w-1));x1=int(np.clip(max(x,px)+pad,x0+1,w))
        y0=int(np.clip(min(y,py)-pad,0,h-1));y1=int(np.clip(max(y,py)+pad,y0+1,h))
        crop=image[y0:y1,x0:x1].copy();del image
        ax.imshow(crop,origin='lower',extent=(x0,x1,y0,y1))
        ax.plot(x,y,'m+',ms=14,mew=2,label='picked')
        ax.plot(px,py,'cx',ms=12,mew=2,label='fit-predicted')
        ax.set(xlim=(x0,x1),ylim=(y0,y1),xticks=[],yticks=[],title=f'{key} (resid={resid:.0f}px)')
        if not (0<=px<w and 0<=py<h):
            ax.text(.02,.02,'Prediction outside image',transform=ax.transAxes,color='red')
    for ax in axes[len(rows):]:ax.axis('off')
    if rows:axes[0].legend(loc='upper right',fontsize=7)
    else:axes[0].text(.5,.5,f'No {landmark} picks with a fitted position',ha='center')
    fig.suptitle(f'{fit_file}: picked vs. fit-predicted {landmark} position')
    fig.tight_layout()
    return fig,axes
