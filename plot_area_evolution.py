import pathlib
import geopandas
import numpy as np
import lmfit
import fit_functions
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


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
def plot_surface_area_evolution(dates,areas):
   """
   Plot the evolution of the lake surface area.
   
   Parameters
   ----------
   dates: list
         month and year of the observations
   area: pandas.core.series.Series
         surface area of the lake in km2

   Returns
   -------
   matplotlib.figure.Figure
         Evolution of the monthly surface area of the lake.
   """

   samp = np.where(areas>np.percentile(areas,5))

   y_min = np.percentile(areas,5)#np.min(areas)
   y_max = np.max(areas)

   ###########################
   ### Add exponential fit ###
   ###########################

   fig,ax = plt.subplots(figsize=(10,7))

   ax.scatter(dates,areas, marker='o', s=8,color='black')
   ax.plot(dates,areas, 'k-')

   model = fit_linear(areas[samp],x=dates[samp])
   area_evo = (model[-1]-model[0])/model[0]*100
   ax.plot(dates[samp],model,'b--',
               label=f'surface area: {area_evo:+.2f}%')
   ax.fill_between(dates,0,np.percentile(areas,5),
         color='xkcd:blue',alpha=0.2,linewidth=0)

   ax.legend(loc='upper right')

   plt.title(f"Lake surface area")
   plt.xlabel("Date")
   plt.ylabel("Surface area (km$^2$)")
   plt.ylim([0.9*y_min,1.1*y_max])

   ax.tick_params(labelright=True,right=True,which='both')
   ax.xaxis.set_major_locator(MultipleLocator(1))
   ax.xaxis.set_minor_locator(MultipleLocator(1/6))
   ax.yaxis.set_major_locator(MultipleLocator(5))
   ax.yaxis.set_minor_locator(MultipleLocator(1))

   fig.tight_layout()

   return fig,area_evo

#--------------------------------------------------------------------


# Paths definition
NOTEBOOK_PATH  = pathlib.Path().resolve()
DATA_DIRECTORY = NOTEBOOK_PATH / "data"
FIG_DIRECTORY  = NOTEBOOK_PATH / "figures"


monthly_record = geopandas.read_file(DATA_DIRECTORY / "lake_patzcuaro.gpkg")

areas = np.array(monthly_record.area)/1e6
dates = np.array(monthly_record.year+(monthly_record.month-1)/12)

fig,area = plot_surface_area_evolution(dates,areas)
#area_evo = np.append(area_evo,f"{area:+.2}")
fig.savefig(FIG_DIRECTORY / f"Surface_area_evolution.png")



