import os
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime,timedelta

import pandas as pd
import geopandas
import rioxarray
from geocube.vector import vectorize
from shapely.geometry import LineString,Polygon
from calendar import monthrange,month_name

import matplotlib.pyplot as plt

from sentinelhub import (
    SHConfig,
    CRS,
    BBox,
    DataCollection,
#    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubCatalog,
#    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)


#--------------------------------------------------------------------
def determine_time_intervals(total_interval,sampling,bbox):
   """
   Determine the time sub-intervals to be used for sentinel requests.
   
   Parameters
   ----------
   time_interval: tuple of str
         time interval covering the period of interest
   sampling: str
         sampling for the data extraction

   Returns
   -------
   list of tuples
        list of the time sub-intervals
   """

   date_b = datetime.strptime(total_interval[0],"%Y-%m-%d")
   date_e = datetime.strptime(total_interval[1],"%Y-%m-%d")
   if(sampling=="monthly"):
      intervals = []
      for year in range(date_b.year,date_e.year,1):
         for month in range(1,13,1):
            last_day = monthrange(year, month)[1]
            date1 = datetime(year,month,1)
            date2 = datetime(year,month,last_day)
            if((date2>date_b)&(date1<date_e)):
               intervals += [(date1.strftime("%Y-%m-%d"),
                              date2.strftime("%Y-%m-%d"))]
   elif(sampling=="weekly"):
      tdelta = timedelta(weeks=1)
      n_bins = int((date_e - date_b)/tdelta)
      edges = [(date_b+i*tdelta).strftime("%Y-%m-%d") for i in range(n_bins+1)]
      intervals = [(edges[i],edges[i + 1]) for i in range(len(edges) - 2)]
      if(date_b+n_bins*tdelta>date_e):
         intervals += [(edges[-2],date_e.strftime("%Y-%m-%d"))]
      else:
         intervals += [(edges[-2],edges[-1])]
         intervals += [(edges[-1],date_e.strftime("%Y-%m-%d"))]
   elif(sampling=="daily"):
      catalog = SentinelHubCatalog(config=config)

      search_iterator = catalog.search(
          DataCollection.SENTINEL2_L1C,
          bbox=bbox,
          time=time_interval,
          fields={"include": ["id", "properties.datetime"], "exclude": []},
      )
      results = list(search_iterator)
      dates = np.unique([res['properties']['datetime'][:10] for res in results])

      intervals = [(date,date) for date in dates]

   return intervals

#--------------------------------------------------------------------
def read_sentinel(config,time_interval,bbox,epsg="EPSG:3067"):
   evalscript_all_bands = """
      //VERSION=3
      function setup() {
         return {
            input: [{
               bands: ["B01","B02","B03","B04","B05","B06","B07",
                       "B08","B8A","B09","B10","B11","B12"],
               units: "DN"
            }],
            output: {
               bands: 13,
               sampleType: "INT16"
            }
         };
      }

      function evaluatePixel(sample) {
         return [sample.B01,
                 sample.B02,
                 sample.B03,
                 sample.B04,
                 sample.B05,
                 sample.B06,
                 sample.B07,
                 sample.B08,
                 sample.B8A,
                 sample.B09,
                 sample.B10,
                 sample.B11,
                 sample.B12];
      }
   """

   request_all_bands = SentinelHubRequest(
      data_folder="tmp_dir",
      evalscript=evalscript_all_bands,
      input_data=[
         SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C.define_from(
               "s2l1c", service_url=config.sh_base_url
            ),
            time_interval=time_interval,
            mosaicking_order=MosaickingOrder.LEAST_CC,
         )
      ],
      responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
      bbox=bbox,
      size=size,
      config=config,
   )


   request_all_bands.save_data()
   source_list = os.listdir('tmp_dir')

   raster = rioxarray.open_rasterio(
               'tmp_dir/'+source_list[0]+'/response.tiff'
            ).rio.reproject(epsg)

   os.system('rm -Rf tmp_dir/'+source_list[0])

   return raster

#--------------------------------------------------------------------
def derive_NDWI(raster):
   """
   Given a Sentinel data, derive the Normalized Difference Water
   Index (NDWI).
   https://custom-scripts.sentinel-hub.com/sentinel-2/ndwi/
   
   Parameters
   ----------
   raster: xarray.core.dataarray.DataArray
         3d raster data from Sentinel

   Returns
   -------
   xarray.core.dataarray.DataArray
        2d raster data of the NDWI
   """

   green = raster.sel(band=3)
   nir   = raster.sel(band=8)
   NDWI = (green-nir)/(green+nir)

   return NDWI

#--------------------------------------------------------------------
def extract_polygon_lake(water,time_interval):
   """
   Given a Sentinel data, derive the Normalized Difference Water
   Index (NDWI).
   
   Parameters
   ----------
   raster: xarray.core.dataarray.DataArray
         3d raster data from Sentinel

   Returns
   -------
   xarray.core.dataarray.DataArray
        2d raster data of the NDWI
   """

   gdf = vectorize(water.astype("float32"))
   gdf = gdf[~np.isnan(gdf["_data"])]
   gdf["area"] = gdf.area
   poly = gdf[gdf["area"] == np.max(gdf["area"])].reset_index()
   poly['date']  = time_interval[0]

   line = poly.geometry.values[0]
   line = LineString(line.exterior)
   d    = {'date': [time_interval[0]], 'geometry': [line]}
   line_gdf = geopandas.GeoDataFrame(d,crs=poly.crs)

   return poly[['date','geometry']],line_gdf

#--------------------------------------------------------------------
def plot_rgb(raster,sampling,date=None):
   """
   Plot the RGB composite associated with the raster image
   after proejction to EPSG:4326.
   
   Parameters
   ----------
   raster: xarray.core.dataarray.DataArray
         3d raster data over the period of interest

   Returns
   -------
   matplotlib.figure.Figure
         RGB composite image of the raster.
   """
   # Reproject to EPSG:4326
   raster = raster.rio.reproject("EPSG:4326")

   # Extract the red, green, and blue bands
   red_band   = raster.sel(band=4)
   green_band = raster.sel(band=3)
   blue_band  = raster.sel(band=2)

   # Stack the bands together to create an RGB composite
   rgb_image = np.dstack((red_band.values, green_band.values, blue_band.values))

   # Normalize the image values between 0 and 1 by dividing by the max value
   rgb_image = rgb_image / np.max(rgb_image)
   rgb_image[np.where(rgb_image<0)] = float('nan')
   rgb_image = np.clip(rgb_image,0,0.45)/0.45


   fig = plt.figure()
   ax = fig.add_subplot()
   im = plt.imshow(rgb_image)

   if(sampling == "monthly"):
      title  = f"RGB composite {month_name[date.month]}"
      title += f" {date.year} (Sentinel-2)"
   else:
      title  = f"RGB composite {date.day} {month_name[date.month]}"
      title += f" {date.year} (Sentinel-2)"
   ax.set_title(title)
#   ax.axis('off')  # Hide the axis for better visualization

   return fig

#--------------------------------------------------------------------
def plot_raster(raster,gdf,sampling,vmin=None,vmax=None,date=None):
   """
   Plot the raster image after projection to EPSG:4326.
   
   Parameters
   ----------
   raster: xarray.core.dataarray.DataArray
         2d raster data over the period of interest
   vmin,vmax: float
         limits of the color bar
   year: int
         year of interest
   month: int
         month of interest

   Returns
   -------
   matplotlib.figure.Figure
         Map of the raster.
   """
   # Reproject to EPSG:4326
   raster = raster.rio.reproject("EPSG:4326")
   gdf = gdf.to_crs("EPSG:4326")

   fig = plt.figure()
   ax = fig.add_subplot()

   im = raster.plot(ax=ax, cmap='terrain', vmin=vmin,vmax=vmax,
            add_colorbar=True)
   gdf.boundary.plot(ax=ax, color='k')

   colorbar = im.colorbar
   if(sampling == "monthly"):
      title = f"{month_name[date.month]} {date.year}"
   else:
      title = f"{date.day} {month_name[date.month]} {date.year}"
   ax.set_title(title)
   ax.set_xlabel("Eastern longitude (degrees)")
   ax.set_ylabel("Latitude (degrees)")

   return fig

#--------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--beginning", type=str,
                    default='2015-06-01',
                    help="Initial date of interest ('YYYY-MM-DD')")
parser.add_argument("-e", "--end",       type=str,
                    default=datetime.now().strftime("%Y-%m-%d"),
                    help="Final date of interest ('YYYY-MM-DD')")
parser.add_argument("-s", "--sampling",  type=str, default='monthly',
                    choices=['daily','weekly','monthly'],
                    help="Sampling for the data extraction.")


args  = parser.parse_args()
time_interval = (args.beginning,args.end)
sampling = args.sampling



# Paths definition
NOTEBOOK_PATH  = pathlib.Path().resolve()
DATA_DIRECTORY = NOTEBOOK_PATH / "data"
FIG_DIRECTORY  = NOTEBOOK_PATH / "figures"


config = SHConfig()

resolution = 60
bbox=BBox((-101.747131,19.711122,-101.503372,19.524847),crs=CRS.WGS84)
size = bbox_to_dimensions(bbox, resolution=resolution)



#raster = read_sentinel(config,
#            time_interval=(dates[0],dates[0]),
#            epsg="EPSG:6369")

#NDWI  = derive_NDWI(raster)

#water = NDWI.where(NDWI > 0)
#water = water/water

#lake,lake_shore = extract_polygon_lake(water)


intervals = determine_time_intervals(time_interval,sampling,bbox)


for interval in tqdm(intervals):
   raster = read_sentinel(config,time_interval=interval,bbox=bbox,
         epsg="EPSG:6369")

   # do not continue if there is no data
   if(raster.max() == 0): continue

   NDWI  = derive_NDWI(raster)

   # do not continue if the data are not good (likely due to clouds)
   if(NDWI.max() < 0.3): continue

   # threshold for water body;
   # following McFeeters (2013): https://doi.org/10.3390/rs5073544 
   water = NDWI.where(NDWI > 0)
   water = water/water

   lake,lake_shore = extract_polygon_lake(water,time_interval=interval)

   try:
      records = pd.concat([records, lake])
   except:
      records = lake.copy()

   if(len(records)<=3): continue
   elif(lake.area[0]<0.85*records.area.median()):
      date = datetime.strptime(interval[0],"%Y-%m-%d")
      rgb = plot_rgb(raster,sampling,date=date)
      if(sampling == "monthly"):
         title = f"RGB_{month_name[date.month]}_{date.year}.png"
      else:
         title = f"RGB_{date.day}_{month_name[date.month]}_{date.year}.png"
      rgb.savefig(FIG_DIRECTORY / title)

      fig = plot_raster(NDWI,lake,sampling,vmin=-1,vmax=1,date=date)
      if(sampling == "monthly"):
         title = f"NDWI_{month_name[date.month]}_{date.year}.png"
      else:
         title = f"NDWI_{date.day}_{month_name[date.month]}_{date.year}.png"
      fig.savefig(FIG_DIRECTORY / title)



# Plots with EPSG:4326

records.to_file(DATA_DIRECTORY / f"lake_patzcuaro_{sampling}.gpkg")



