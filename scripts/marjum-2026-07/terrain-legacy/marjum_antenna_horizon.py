"""Export DEM-relative antenna horizons; candidate spread is NOT a posterior."""
from pathlib import Path
import argparse
import json
import numpy as np
from marjum_position import RefinedTerrain
from marjum_grid import GridCoordinates
from marjum_mcmc import digest


def source_terrain(terrain,grid):
    """Read fractional-meter source heights without changing the DEM cache."""
    from PIL import Image
    result=RefinedTerrain(terrain.dem)
    result.data=np.full(terrain.data.shape,np.nan,dtype=np.float32)
    for filename in np.asarray(terrain.dem.files).ravel():
        with Image.open(str(filename)) as im:
            tie=im.tag_v2[33922];scale=im.tag_v2[33550]
            xy=np.array([tie[3]+.5*scale[0],tie[4]-(im.height-.5)*scale[1]])-grid.origin
            col=int(round((xy[0]-terrain.e[0])/terrain.res));row=int(round((xy[1]-terrain.n[0])/terrain.res))
            if col<0 or row<0 or row+im.height>result.data.shape[0] or col+im.width>result.data.shape[1]:
                raise ValueError('Source tile does not align with cached DEM extent')
            result.data[row:row+im.height,col:col+im.width]=np.flipud(np.asarray(im,dtype=np.float32))
    if not np.isfinite(result.data).all():raise ValueError('Source DEM coverage is incomplete')
    return result


def export(output='cv_position_absolute',stages=('baseline','poses','alternative','focal')):
    from eigsep_terrain.marjum_dem import MarjumDEM
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(output);out=root/'antenna_horizon'
    out.mkdir(exist_ok=False)
    terrain=RefinedTerrain(MarjumDEM(cache_file='marjum_dem_sw.npz'));grid=GridCoordinates(terrain.dem)
    bearing=np.arange(3600)/10.;states={s:dict(np.load(root/f'{s}.npz')) for s in stages}
    report=dict(note='Finite-DEM terrain-only horizon. Candidate differences are model/start sensitivity, not credible intervals. Vertical datum is unverified; no 2025 antenna constraint.',
        grid_epsg=grid.epsg,grid_origin_projected_m=grid.origin.tolist(),candidates={},
        input_sha256={str(p):digest(p) for p in [Path(__file__),Path('marjum_position.py'),Path('marjum_grid.py'),Path('marjum_dem_sw.npz')]+[root/f'{s}.npz' for s in stages]})
    curves={};fig,ax=plt.subplots(figsize=(12,4))
    for label,state in states.items():
        ant=state['antenna'];north=grid.true_north_grid_bearing(*ant[:2])
        # Ray routines use mathematical azimuth (east=0, north=90 degrees).
        az=np.deg2rad(90-bearing-north)
        elev=np.rad2deg(np.concatenate([terrain.skyline(ant,a) for a in np.array_split(az,30)]))
        curves[label]=elev;lon,lat=grid.to_lonlat(*ant[:2])
        report['candidates'][label]=dict(antenna_grid_m=ant.tolist(),antenna_projected_en_m=(ant[:2]+grid.origin).tolist(),
            approximate_lon_lat_deg=[float(lon),float(lat)],height_above_dem_m=float(ant[2]-terrain.height(*ant[:2])),
            true_north_grid_bearing_deg=north)
        np.savetxt(out/f'{label}.csv',np.c_[bearing,elev],delimiter=',',header='true_bearing_deg,elevation_deg',comments='')
        ax.plot(bearing,elev,label=label,lw=1)
    ref=curves[stages[-1]]
    for label,elev in curves.items():
        report['candidates'][label]['difference_from_last_candidate_deg']=dict(rms=float(np.sqrt(np.mean((elev-ref)**2))),max_abs=float(abs(elev-ref).max()))
    # Hold antenna coordinates fixed to isolate DEM quantization sensitivity.
    native=source_terrain(terrain,grid);ant=states[stages[-1]]['antenna']
    az=np.deg2rad(90-bearing-grid.true_north_grid_bearing(*ant[:2]))
    elev=np.rad2deg(np.concatenate([native.skyline(ant,a) for a in np.array_split(az,30)]))
    report['source_dem_fixed_position_check']=dict(rms_horizon_change_deg=float(np.sqrt(np.mean((elev-ref)**2))),
        max_abs_horizon_change_deg=float(abs(elev-ref).max()),height_above_source_dem_m=float(ant[2]-native.height(*ant[:2])),
        note='Antenna coordinates are held fixed: this is not a refit to fractional-meter source elevations.')
    curves['focal_source_dem_fixed_position']=elev
    np.savetxt(out/'source_dem_fixed_position.csv',np.c_[bearing,elev],delimiter=',',header='true_bearing_deg,elevation_deg',comments='')
    ax.plot(bearing,elev,label='source DEM, fixed position',lw=.8,ls='--')
    ax.set(xlabel='True compass bearing (degrees)',ylabel='DEM horizon elevation (degrees)',xlim=(0,360),title='Antenna terrain horizon — candidate sensitivity, not posterior uncertainty')
    ax.legend();fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    np.savez_compressed(out/'horizons.npz',true_bearing_deg=bearing,**curves)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['candidates'],indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='cv_position_absolute');p.add_argument('--stages',nargs='+',default=['baseline','poses','alternative','focal'])
    export(**vars(p.parse_args()))
