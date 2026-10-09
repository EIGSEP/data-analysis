"""Native GeoTIFF grid coordinates without the legacy survey-offset mapping."""
from pathlib import Path
import numpy as np
from PIL import Image
from pyproj import Transformer


class GridCoordinates:
    def __init__(self,dem):
        tiles=[]
        for filename in np.asarray(dem.files).ravel():
            with Image.open(str(filename)) as im:
                scale=im.tag_v2[33550];tie=im.tag_v2[33922];raw=im.tag_v2[34735]
                tags={raw[j]:raw[j+3] for j in range(4,len(raw),4)}
                if tags.get(1025)!=1:raise ValueError('Expected PixelIsArea GeoTIFFs')
                if not np.allclose(scale[:2],dem.res):raise ValueError('Raster resolutions disagree')
                x0=tie[3]-tie[0]*scale[0];y0=tie[4]+tie[1]*scale[1]-im.height*scale[1]
                tiles.append((x0,y0,im.width,im.height,tags[3072]))
        codes={v[4] for v in tiles}
        if len(codes)!=1:raise ValueError('Mixed projected CRSs')
        self.epsg=int(codes.pop());self.res=float(dem.res)
        self.origin=np.array([min(v[0] for v in tiles),min(v[1] for v in tiles)])+.5*self.res+np.array([dem.e0_px,dem.n0_px])*self.res
        self.forward=Transformer.from_crs(4326,self.epsg,always_xy=True)
        self.inverse=Transformer.from_crs(self.epsg,4326,always_xy=True)
        self.legacy_survey_offset=np.asarray(dem.survey_offset).tolist()

    def from_lonlat(self,lon,lat):
        e,n=self.forward.transform(lon,lat)
        return np.stack([e,n],axis=-1)-self.origin

    def to_lonlat(self,e,n):
        return self.inverse.transform(e+self.origin[0],n+self.origin[1])

    def true_north_grid_bearing(self,e,n):
        lon,lat=self.to_lonlat(e,n)
        delta=self.from_lonlat(lon,lat+1e-5)-self.from_lonlat(lon,lat)
        return float(np.rad2deg(np.arctan2(delta[0],delta[1])))

    def gps(self,keys):
        from eigsep_terrain.exif import read_exif
        rows=[read_exif(f'marjum-2026-07/IMG_{k}.HEIC') for k in keys]
        xy=np.array([self.from_lonlat(v['lon'],v['lat']) for v in rows])
        alt=np.array([v['alt'] for v in rows],float)
        if not np.isfinite(xy).all() or not np.isfinite(alt).all():raise ValueError('Missing GPS metadata')
        return xy,alt,rows
