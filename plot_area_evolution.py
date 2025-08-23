import json
import pathlib
import pandas as pd
import geopandas
import numpy as np
import lmfit
import fit_functions
import calendar
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


#--------------------------------------------------------------------
def convert_date2float(date_str):
   """
   Read the date as a string and convert it into a float
   that can be used to plot and fit the data.
   
   Parameters
   ----------
   date_str: str
         date in string format (%Y-%m-%d)

   Returns
   -------
   numpy.array of float
   """
   date      = datetime.strptime(date_str,"%Y-%m-%d")
   timetuple = date.timetuple()

   if(calendar.isleap(timetuple.tm_year)):
      fdate = timetuple.tm_year+(timetuple.tm_yday-1)/366
   else:
      fdate = timetuple.tm_year+(timetuple.tm_yday-1)/365

   return fdate

#--------------------------------------------------------------------
def fit_linear(y,x=None,err=None,method='leastsq'):
   """
   Minimize the data points with a linear function (linear regression)
   
   Parameters
   ----------
   y: 1D array
         data points of the dependant variable
   x: 1D array
         data points of the independant variable
   err: 1D array
         errors on the dependant variable
   method: str
         minimization method

   Returns
   -------
   1D array
         data points of the results of the linear regression
   """
   if x is None:
      x = np.arange(len(y))

   fit_params = lmfit.create_params(
            coeff={'value': -2, 'min': -20, 'max': -0.2},
            const={'value': np.mean(y), 'min': 60, 'max': 2000})

   fitter = lmfit.Minimizer(fit_functions.linear,
               fit_params,fcn_args=(x,y,err))
   try:
      fit = fitter.minimize(method=method)
   except Exception as mes:
      print("Something wrong with fit: ", mes)
      raise SystemExit

   model = fit_functions.linear(fit.params,x)

   return model

#--------------------------------------------------------------------
def plot_surface_area_evolution(areas,lake_name=""):
   """
   Plot the evolution of the lake surface area.
   
   Parameters
   ----------
   areas: geopandas.geodataframe.GeoDataFrame
         date, monthly, weekly and daily records of the surface area
   lake_name: str
         name of the lake

   Returns
   -------
   matplotlib.figure.Figure
         Evolution of the monthly surface area of the lake.
   """
   #areas = areas.sort_values(by=['float_date'])

   # Read the temperature for the selected city
   dates   = np.array(areas["float_date"]).astype(float)
   daily   = np.array(areas["area_daily"]).astype(float)
   weekly  = np.array(areas["area_weekly"]).astype(float)
   monthly = np.array(areas["area_monthly"]).astype(float)

   y_min   = np.nanpercentile([daily,weekly,monthly],5)
   y_max   = np.nanmax([daily,weekly,monthly])



   fig,ax = plt.subplots(figsize=(10,7))

   ### Plot monthly records
   samp = np.where(np.isfinite(monthly))
   if(len(samp[0]>0)):
      ax.scatter(dates,monthly, marker='o', s=8,color='black')
      ax.plot(dates,monthly, 'k-')

      samp = np.where((np.isfinite(monthly))&
                      (monthly>np.nanpercentile(monthly,5))&
                      (monthly<np.nanpercentile(monthly,95)))
      model = fit_linear(monthly[samp],x=dates[samp])
      area_evo = (model[-1]-model[0])/model[0]*100
      ax.plot(dates[samp],model,'k--',
                  label=f'surface area: {area_evo:+.2f}%')


   ### Plot weekly records
   samp = np.where(np.isfinite(weekly))
   if(len(samp[0]>0)):
      ax.scatter(dates,weekly, marker='o', s=8,color='red')
      ax.plot(dates,weekly, 'r-')

      samp = np.where((np.isfinite(weekly))&
                      (weekly>np.nanpercentile(weekly,5))&
                      (weekly<np.nanpercentile(weekly,95)))
      model = fit_linear(weekly[samp],x=dates[samp])
      area_evo = (model[-1]-model[0])/model[0]*100
      ax.plot(dates[samp],model,'r--',
                  label=f'surface area: {area_evo:+.2f}%')


   ### Plot daily records
   samp = np.where(np.isfinite(daily))
   if(len(samp[0]>0)):
      ax.scatter(dates,daily, marker='o', s=8,color='blue')
      ax.plot(dates,daily, 'b-')

      samp = np.where((np.isfinite(daily))&
                      (daily>np.nanpercentile(daily,5))&
                      (daily<np.nanpercentile(daily,95)))
      model = fit_linear(daily[samp],x=dates[samp])
      area_evo = (model[-1]-model[0])/model[0]*100
      ax.plot(dates[samp],model,'b--',
                  label=f'surface area: {area_evo:+.2f}%')

   ax.legend(loc='upper right')

   plt.title(f"Lake {lake_name} surface area")
   plt.xlabel("Date")
   plt.ylabel("Surface area (km$^2$)")
   plt.ylim([0.9*y_min,1.1*y_max])

   ax.tick_params(labelright=True,right=True,which='both')
   ax.xaxis.set_major_locator(MultipleLocator(1))
   ax.xaxis.set_minor_locator(MultipleLocator(1/6))
#   ax.yaxis.set_major_locator(MultipleLocator(5))
#   ax.yaxis.set_minor_locator(MultipleLocator(1))

   fig.tight_layout()

   return fig,area_evo

#--------------------------------------------------------------------


# Paths definition
NOTEBOOK_PATH  = pathlib.Path().resolve()
DATA_DIRECTORY = NOTEBOOK_PATH / "data"
FIG_DIRECTORY  = NOTEBOOK_PATH / "figures"


with open(DATA_DIRECTORY / "lakes_mexico_catalogue.json") as file:
   dict = json.load(file)


area_evo = np.array([])
for lake_name in ["Patzcuaro"]:#dict:
   areas = pd.DataFrame(columns=["date","float_date"]+[f"area_{sampling}"
               for sampling in ["daily","weekly","monthly"]])

   for sampling in ["monthly"]:
      records   = geopandas.read_file(DATA_DIRECTORY / 
                              f"{lake_name}_lake_evolution.gpkg",
                              layer=f"{sampling}")

      for i in range(len(records)):
         entry = records.iloc[i]
         d = {"date": entry['date'],
              "float_date": convert_date2float(entry['date'])
             }
         if(sampling=="daily"):
            d[f"area_daily"]   = [entry['geometry'].area/1e6]
            d[f"area_weekly"]  = [np.nan]
            d[f"area_monthly"] = [np.nan]
         elif(sampling=="weekly"):
            d[f"area_daily"]   = [np.nan]
            d[f"area_weekly"]  = [entry['geometry'].area/1e6]
            d[f"area_monthly"] = [np.nan]
         elif(sampling=="monthly"):
            d[f"area_daily"]   = [np.nan]
            d[f"area_weekly"]  = [np.nan]
            d[f"area_monthly"] = [entry['geometry'].area/1e6]

         new_row = pd.DataFrame(d)
         areas   = pd.concat([areas, new_row], ignore_index=True)

      if(sampling == "monthly"):
         # Select the entry with the larger area (95th-percentile)
         perc95 = np.percentile(records.area,95)
         diff   = np.absolute(records.area-perc95)
         lake   = records[diff == np.min(diff)].reset_index()
         lake["LAKE_NAME"] = lake_name
         lake["AREA_KM2"] = lake.area/1e6

         try:
            lakes = pd.concat([lakes,lake[['LAKE_NAME','AREA_KM2','geometry']]],
                  ignore_index=True)
         except:
            lakes = lake[['LAKE_NAME','AREA_KM2','geometry']].copy()

   fig,area = plot_surface_area_evolution(areas,lake_name=lake_name)
   fig.savefig(FIG_DIRECTORY / f"Surface_area_evolution_{lake_name}.png")

   area_evo = np.append(area_evo,f"{area:+.2}")

lakes["AREA_EVO"] = area_evo
lakes.to_file(DATA_DIRECTORY /
      "lakes_surface_area_evolution.gpkg")


