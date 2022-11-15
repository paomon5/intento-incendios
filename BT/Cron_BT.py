#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os, glob
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xrc
import datetime as dt
import gc
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point
import shapely.geometry as ss
import matplotlib.pyplot as plt
import BT_functions as btf

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import sys

fecha = dt.datetime.today().date()#.strftime("%Y-%m-%d")
lati =   6.25
loni = -75.60
source = 'GFS'
## GENERACIÓN DE BT HACIA ATRÁS,  ESTAS SON LAS QUE SE USAN PARA LA ROSA DE VIENTOS

delta_t = 3  # horas
leveli  = 800 # hPa   
path_files = '/var/meteo/GFS/processed/'
path_out   = '/var/meteo/GFS/BT/BT_Aire/'
ndays = 5

# Variables extraidas de los datos
files_date_2, files_list_2, lat_dataset, lon_dataset, levels_dataset = \
     btf.Follow(path=path_files, path_fig=None, source=source,
     lati=lati,loni=loni, warnings=False)

hoy = fecha
date_name = f'{hoy.strftime("%Y%m%d")}'
arch_hoy = np.sort(glob.glob(path_files + "*"+date_name+"*"))
ult_fecha = arch_hoy[-1].split("/")[-1][-13:-3]
fecha_dt = dt.datetime.strptime(ult_fecha, "%Y%m%d%H")


for i, date_i in enumerate([fecha_dt]):
    print("GENERANDO BT DEL ", date_i)
    # Nombre del archivo, correspondiente al día evaluado
    date_name = f'{date_i.strftime("%Y%m%d%H")}'
    arch_viento_actual = f"gfs_t00z_pgrb2_{date_name}.nc"
    aa = np.where(files_list_2<= path_files + arch_viento_actual)[0]
    files_list = files_list_2[aa]
    files_date = files_date_2[aa]

    Fechai = (date_i).strftime('%Y-%m-%d %H:00')
    Fechaf = (date_i + dt.timedelta(hours = 23)).strftime('%Y-%m-%d %H:00')#esto es para generar las bt de todo ese día
    print(f'{Fechai} -----> {Fechaf}')
    name_file = f'BT_GFS.{delta_t}h.{leveli}hPa.{date_name}.{ndays}days.nc'


    # Calculamos la retrotrayectoria para un solo día
    BT = btf.Trajectories_level_i(lati=lati,loni=loni,leveli=leveli,Fechai=Fechai, Fechaf=Fechaf, delta_t=delta_t,
        ndays=ndays, source='GFS',path_files=path_files,
        files_date=files_date, files_list=files_list, 
        lat_dataset=lat_dataset, lon_dataset=lon_dataset, 
        levels_dataset=levels_dataset)

    # Se almacena el nuevo archivo  NetCDF4 
    btf.save_nc(dictionary=BT, file_out=f'{path_out}{name_file}')
    del(BT)
    gc.collect()





# Propiedades del cálculo
delta_t = 1  # horas
leveli  = 800 # hPa    

path_files = '/var/meteo/GFS/processed/'
path_out   = '/var/meteo/GFS/BT/'

ndays = 5

# Variables extraidas de los datos
files_date_2, files_list_2, lat_dataset, lon_dataset, levels_dataset = \
     btf.Follow(path=path_files, path_fig=None, source=source,
     lati=lati,loni=loni, warnings=False)

FECHAS = pd.date_range("2022-10-27", "2022-11-09")
for fecha in [fecha]:
    hoy = fecha
    date_name = f'{hoy.strftime("%Y%m%d")}'
    arch_hoy = np.sort(glob.glob(path_files + "*"+date_name+"*"))
    ult_fecha = arch_hoy[-1].split("/")[-1][-13:-3]
    fecha_dt = dt.datetime.strptime(ult_fecha, "%Y%m%d%H")


    for i, date_i in enumerate([fecha_dt]):
        print("GENERANDO BT DEL ", date_i)
        # Nombre del archivo, correspondiente al día evaluado
        date_name = f'{date_i.strftime("%Y%m%d%H")}'
        arch_viento_actual = f"gfs_t00z_pgrb2_{date_name}.nc"
        aa = np.where(files_list_2<= path_files + arch_viento_actual)[0]
        files_list = files_list_2[aa]
        files_date = files_date_2[aa]
    #        print("shapeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", files_date.shape)


        Fechai = date_i.strftime('%Y-%m-%d %H:00')
        Fechaf = (date_i+dt.timedelta(hours = 115)).strftime('%Y-%m-%d %H:00') ## Este 115 es por las 115 horas del pronóstico
        print(f'{Fechai} -----> {Fechaf}')
        name_file = f'BT_GFS.{delta_t}h.{leveli}hPa.{date_name}.{ndays}days.nc'


        # Calculamos la retrotrayectoria para un solo día
        BT = btf.Trajectories_level_i(lati=lati,loni=loni,leveli=leveli,Fechai=Fechai, Fechaf=Fechaf, delta_t=delta_t,
            ndays=ndays, source='GFS',path_files=path_files,
            files_date=files_date, files_list=files_list, 
            lat_dataset=lat_dataset, lon_dataset=lon_dataset, 
            levels_dataset=levels_dataset)

        # Se almacena el nuevo archivo  NetCDF4 
        btf.save_nc(dictionary=BT, file_out=f'{path_out}{name_file}')
        del(BT)
        gc.collect()