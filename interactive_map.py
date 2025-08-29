import pathlib
import json
import pandas as pd
import geopandas
import numpy as np

import folium
import base64
import branca.colormap
from geocube.vector import vectorize
from folium.plugins import GroupedLayerControl

from calendar import month_name
#from geoanalysis_functions import extract_temperatures


from branca.element import MacroElement
from jinja2 import Template


#--------------------------------------------------------------------
def lakes_with_country():
   """
   Produce a new layer based on raster data
   
   Parameters
   ----------
   raster: xarray.core.dataarray.DataArray
         2d raster data to include
   name: str
         Name of the layer
   visible: boolean
         Define if the layer will be displayed when opening the map
   year: int
         year to be added in name
   month: int
         month to be added in name
   day: int
         day to be added in name

   Returns
   -------
   layer: folium.features.GeoJson
         layer to be added to the interactive map
   cmap: branca.colormap.LinearColormap
         corresponding color map
   """

   countries = geopandas.read_file(DATA_DIRECTORY /
         "ne_10m_admin_0_countries")
   lakes     = geopandas.read_file(DATA_DIRECTORY /
         "lakes_surface_area_evolution.gpkg")

   # Determine the country where the lake is located
   poly = lakes.to_crs(countries.crs)
   assert poly.crs == countries.crs, "CRS are not identical"
   poly_with_country = poly.sjoin(countries,how="left",predicate="within")
   lakes["COUNTRY"] = poly_with_country["NAME"]

   return lakes[['LAKE_NAME','COUNTRY','AREA_KM2','geometry']]

#--------------------------------------------------------------------
class BindColormap(MacroElement):
    """Binds a colormap to a given layer.

    Parameters
    ----------
    colormap : branca.colormap.ColorMap
        The colormap to bind.
    """
    def __init__(self, layer, colormap):
        super(BindColormap, self).__init__()
        self.layer = layer
        self.colormap = colormap
        self._template = Template(u"""
        {% macro script(this, kwargs) %}
            {{this.colormap.get_name()}}.svg[0][0].style.display = 'block';
            {{this._parent.get_name()}}.on('overlayadd', function (eventLayer) {
                if (eventLayer.layer == {{this.layer.get_name()}}) {
                    {{this.colormap.get_name()}}.svg[0][0].style.display = 'block';
                }});
            {{this._parent.get_name()}}.on('overlayremove', function (eventLayer) {
                if (eventLayer.layer == {{this.layer.get_name()}}) {
                    {{this.colormap.get_name()}}.svg[0][0].style.display = 'none';
                }});
        {% endmacro %}
        """)

#--------------------------------------------------------------------
def new_image_layer(raster,name="",visible=False,
         year=2024,month=1,day=None):
   """
   Produce a new layer based on raster data
   
   Parameters
   ----------
   raster: xarray.core.dataarray.DataArray
         2d raster data to include
   name: str
         Name of the layer
   visible: boolean
         Define if the layer will be displayed when opening the map
   year: int
         year to be added in name
   month: int
         month to be added in name
   day: int
         day to be added in name

   Returns
   -------
   layer: folium.features.GeoJson
         layer to be added to the interactive map
   cmap: branca.colormap.LinearColormap
         corresponding color map
   """

   raster = raster.rio.reproject("EPSG:4326")
   raster = raster.rename("raster")
   gdf    = vectorize(raster.astype("float32"))
   gdf["id"] = gdf.index.astype(str)


   if(month==1):
      colors=branca.colormap.linear.Blues_05.colors
      invert=[colors[-i-1] for i in range(len(colors))]
      cmap = branca.colormap.LinearColormap(
            colors=invert,
            vmin=-15,vmax=0,
            caption="Mean temperature")
      fill_color="Blues_r"
   else:
      cmap = branca.colormap.LinearColormap(
            colors=branca.colormap.linear.YlOrRd_08.colors,
            vmin=10,vmax=20,
            caption="Mean temperature")
      fill_color="YlOrRd"

#   layer = folium.Choropleth(
#         geo_data=gdf,
#         data=gdf,
#         columns=("id","_data"),
#         key_on="feature.id",

#         bins=10,
#         fill_color=fill_color,
##         colormap=cmap,
#         line_weight=0,
#         opacity=0.6,
#         legend_name="Mean temperature",
#         name=name,
#         show=visible,

##         highlight=True
#   )

   temp_dict = gdf.set_index('id')['raster']

   layer = folium.GeoJson(
      gdf,
      style_function=lambda feature: {
            "fillColor": cmap(temp_dict[feature['id']]),
            "color": "black",
            "weight": 0,
            "fillOpacity": 0.6,
            },
      name=name,
      show=visible
   )

   for child in layer._children:
        if child.startswith("color_map"):
            del layer._children[child]

   return layer,cmap

#--------------------------------------------------------------------
def new_polygon_layer(gdf,name="",image=None):
   """
   Produce a polygon layer based on a GeoDataFrame

   Parameters
   ----------
   gdf: geopandas.geodataframe.GeoDataFrame
         Municipalities of interest with temperatures information
   name: str
         Name of the layer
   image: str
         Path of the image to be included in a popup
   lake_name: str
         name of the lake to be added in image name

   Returns
   -------
   folium.features.GeoJson
         layer to be added to the interactive map
   """

   if image is not None:
      popup_html = np.array([])
      for idx in range(len(gdf)):
         lake_name = gdf["LAKE_NAME"][idx]
         imname    = image.name.format(lake_name)
         try:
            encoded = base64.b64encode(
                        open(image.with_name(imname), 'rb').read()).decode()
            string = f'<img src="data:image/png;base64,{encoded}" width="300">'
         except:
            string = ''
         popup_html = np.append(popup_html,string)
      gdf['popup_html'] = popup_html
      gdf = gdf[gdf['popup_html'] != ""]

      popup = folium.GeoJsonPopup(
         fields=["popup_html"],
         aliases=[""],
         labels=True,
         sticky=False,
         localize=True,
      )
   else:
      popup=None

   # Define custom tooltip with HTML
   tooltip = folium.features.GeoJsonTooltip(
      fields=("LAKE_NAME",),#,f"SURFACE_EVO",),
      aliases=("Lake:",),#"2016-2025 (km$^2$):",),
      labels=True,
      sticky=False,
      localize=True,
      )

   layer = folium.GeoJson(
      gdf,
      style_function=lambda feature: {
            "fillColor": "transparent",
            "color": "black",
            "weight": 2
            },
      opacity=0.8,
      name=name,
      show=True,
      tooltip=tooltip,
      popup=popup
   )

   return layer

#--------------------------------------------------------------------



# Paths definition
NOTEBOOK_PATH  = pathlib.Path().resolve()
DATA_DIRECTORY = NOTEBOOK_PATH / "data"
FIG_DIRECTORY  = NOTEBOOK_PATH / "figures"
HTML_DIRECTORY = NOTEBOOK_PATH / "html"


with open(DATA_DIRECTORY / "lakes_mexico_catalogue.json") as file:
   dict = json.load(file)


lakes = lakes_with_country()


#copyright  = 'Temperature data (c) <a href="https://en.ilmatieteenlaitos.fi/">'
#copyright += 'Finnish Meteorological Institute</a> & '
#copyright += '<a href="https://paituli.csc.fi/download.html">Paituli</a>, '
#copyright += 'Map data (c) <a href="http://www.openstreetmap.org/copyright">'
#copyright += 'OpenStreetMap</a> contributors.'

# Initial map
interactive_map = folium.Map(
#    max_bounds=True,
    location=[19.7,-101.3],
    zoom_start=10,
    min_lat=19.4,
    max_lat=20.2,
    min_lon=-101.9,
    max_lon=-100.6,
    interactive=True,
#    attr = copyright
)


imname = FIG_DIRECTORY
imname = imname / "Surface_area_evolution_{}.png"
folium_layer = new_polygon_layer(lakes,"Lakes",
         image=imname)

folium_layer.add_to(interactive_map)


interactive_map.save(HTML_DIRECTORY /
            "map_lakes_mexico.html")


