import pathlib
import datetime
import os
import getpass
from tqdm import tqdm

import pandas as pd
import geopandas
import rioxarray
from geocube.vector import vectorize
from shapely.geometry import LineString,Polygon
from calendar import monthrange
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from sentinelhub import (
    SHConfig,
    CRS,
    BBox,
    DataCollection,
#    DownloadRequest,
    MimeType,
    MosaickingOrder,
#    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)


#--------------------------------------------------------------------
def determine_time_interval(year,month):
   """
   Determine the time interval corresponding to the given month.
   The time interval is formatted to be used for sentinel requests.
   
   Parameters
   ----------
   year: int
         year of interest
   month: int
         month of interest

   Returns
   -------
   tuple of str
        time interval that cover the full month
   """

   last_day = monthrange(year, month)[1]
   date1 = datetime(year,month,1).isoformat()[:10]
   date2 = datetime(year,month,last_day).isoformat()[:10]

   return (date1,date2)

#--------------------------------------------------------------------
def read_sentinel(config,time_interval,epsg="EPSG:3067"):
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
def extract_polygon_lake(water,year=2024,month=1):
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
   gdf["area"]   = gdf.area
   poly = gdf[gdf["area"] == np.max(gdf["area"])].reset_index()
   poly['year']  = year
   poly['month'] = month

   line = poly.geometry.values[0]
   line = LineString(line.exterior)
   d    = {'year': [year], 'month': [month], 'geometry': [line]}
   line_gdf = geopandas.GeoDataFrame(d,crs=poly.crs)

   return poly[['year','month','geometry']],line_gdf

#--------------------------------------------------------------------


# Paths definition
NOTEBOOK_PATH  = pathlib.Path().resolve()
DATA_DIRECTORY = NOTEBOOK_PATH / "data"


config = SHConfig()

resolution = 60
bbox=BBox((-101.747131,19.711122,-101.503372,19.524847),crs=CRS.WGS84)
size = bbox_to_dimensions(bbox, resolution=resolution)



#raster = read_sentinel(config,
#            time_interval=("2015-09-01", "2015-09-30"),
#            epsg="EPSG:6369")


#NDWI  = derive_NDWI(raster)

#water = NDWI.where(NDWI > 0)
#water = water/water

#lake,lake_shore = extract_polygon_lake(water)


for year in tqdm(range(2015,2026,1)):
   for month in range(1,13,1):
      time_interval = determine_time_interval(year,month)
      raster = read_sentinel(config,time_interval=time_interval,
            epsg="EPSG:6369")

      # do not continue if there is no data
      if(raster.max() == 0): continue

      NDWI  = derive_NDWI(raster)

      # do not continue if the data are not good (likely due to clouds)
      if(NDWI.max() < 0.25): continue

      water = NDWI.where(NDWI > 0)
      water = water/water

      lake,lake_shore = extract_polygon_lake(water,year,month)

      try:
         monthly_record = pd.concat([monthly_record, lake])
      except:
         monthly_record = lake.copy()

# Plots with EPSG:4326

monthly_record.to_file(DATA_DIRECTORY / "lake_patzcuaro.gpkg")



