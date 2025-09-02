# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 15:58:29 2025

@author: jorgh
"""
import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1) Carga de presencia
csv_pres = r"C:\JORGE_HERRERA\PHD\curso_ing_ambiental_UC\observations-608227.csv"   # <--- tu CSV con columnas 'lat' y 'lon'
df = pd.read_csv(csv_pres)
gdf_pres = gpd.GeoDataFrame(df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326")
gdf_pres.head()

#%% #elimino registros duplicados
gdf_pres = gpd.GeoDataFrame(gdf_pres,
    geometry=gpd.points_from_xy(gdf_pres.longitude, gdf_pres.latitude),
    crs="EPSG:4326")
len(gdf_pres)
# Cambia el directorio actual del proceso Python
os.chdir(r"C:\JORGE_HERRERA\PHD\curso_ing_ambiental_UC")
#%%# 2) Lista de ráster (suelo + bioclim)
soil_folder    = "SoilMaps_MEAN"
bioclim_folder = "Bioclim_1959_2025"
# Listar todos los .tif de ambas carpetas
soil_rasters    = glob.glob(os.path.join(soil_folder,    "*.tif"))
#soil_rasters = [p for p in soil_rasters if "coverage_mapbiomas" not in p]
bioclim_rasters = glob.glob(os.path.join(bioclim_folder, "*.tif"))

# Unir ambas listas
rasters = soil_rasters + bioclim_rasters

# 3) Función para extraer valores ráster en cada punto
def extract_raster_values(gdf, raster_paths):
    for rp in raster_paths:
        name = os.path.splitext(os.path.basename(rp))[0]
        with rasterio.open(rp) as src:
            coords = [(pt.x, pt.y) for pt in gdf.geometry]
            vals = [v[0] for v in src.sample(coords)]
        gdf[name] = vals
    return gdf

gdf_pres = extract_raster_values(gdf_pres, rasters)
gdf_pres["presence"] = 1

# 4) Generar pseudo-ausencias
#    Usamos la extensión del primer ráster como área de muestreo
with rasterio.open(rasters[0]) as ref:
    minx, miny, maxx, maxy = ref.bounds
    transform = ref.transform
    nodata = ref.nodata

n = len(gdf_pres)
abs_pts = []
abs_vals = []

# 1) Abre el raster de referencia **una única vez** y extrae todo lo que necesitas
ref_path = rasters[0]
with rasterio.open(ref_path) as ref:
    minx, miny, maxx, maxy = ref.bounds
    transform = ref.transform
    nodata = ref.nodata
    # Cargamos el array completo de la banda 1
    ref_data = ref.read(1)
    height, width = ref_data.shape
gdf_pres.head(10)

#%% 1) Cargar el shapefile o archivo GeoPackage con puntos
gdf_abs = gpd.read_file("pseudo_ausencia2.shp")  # o .gpkg

# 2) Asegurar que está en el mismo CRS que los rásters (por ejemplo EPSG:4326)
gdf_abs = gdf_abs.to_crs("EPSG:4326")
print(len(gdf_abs))
gdf_abs.head(10)

#%%# 2) Ahora generamos los puntos de pseudo-ausencia usando ese array en memoria

gdf_abs = extract_raster_values(gdf_abs, rasters)
gdf_abs["presence"] = 0
gdf_abs
predictores = [os.path.splitext(os.path.basename(r))[0] for r in rasters]
predictores

#%%# 5) Combinar y limpiar
gdf_all = pd.concat([gdf_pres, gdf_abs], ignore_index=True)
# quitar filas con cualquier NA
gdf_all = gdf_all.dropna(subset=[os.path.splitext(os.path.basename(r))[0] for r in rasters])

#%%# 1. Marca las filas que cumplen ambas condiciones:
mask = (
    gdf_all.duplicated(
        subset=predictores,  keep=False
    )
) & (gdf_all['presence'] == 1)

# 2. Filtra eliminando esas filas:
gdf_filtrado = gdf_all.loc[~mask].copy()
len(gdf_filtrado)
gdf_all['presence'].value_counts()
#%%# 6) Preparar X, y y entrenar Random Forest
feature_cols = [os.path.splitext(os.path.basename(r))[0] for r in rasters]
X = gdf_filtrado[feature_cols]
X = pd.get_dummies(
    X,
    columns=["coverage_mapbiomas"],
    prefix="mapbio_",
    drop_first=True,
    dtype=np.uint8
)
y = gdf_filtrado["presence"]
X
#%% Entrenamos un modelo para seleccionar numero de variables
import numpy as np
import pandas as pd
import mrmr   # pip install mrmr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix)

# 3) Loop mRMR + entrenamiento/evaluación
K_list = np.arange(2, min(40, X.shape[1]+1))
results = []

for k in K_list:
    print(k)
    # 3.1) Selección mRMR para clasificación
    selected = mrmr.mrmr_classif(X=X, y=y, K=k)
    
    # 3.2) Dataset con solo las k variables + y
    df_k = pd.concat([y, X[selected]], axis=1).dropna()
    Xk, yk = df_k[selected], df_k["presence"]
    
    # 3.3) División train/test
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xk, yk, test_size=0.3, stratify=yk, random_state=42
    )
    
    # 3.4) Entrena un Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:,1]
    
    # 3.5) Métricas
    acc  = accuracy_score(y_te, y_pred)
    bacc = balanced_accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec  = recall_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred)
    auc  = roc_auc_score(y_te, y_proba)
    
    results.append({
        "K": k,
        "accuracy": acc,
        "balanced_acc": bacc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc
    })

results_df = pd.DataFrame(results)
print(results_df)

# 4) Graficar métricas vs K
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
for metric in ["accuracy","balanced_acc","f1","roc_auc"]:
    plt.plot(results_df["K"], results_df[metric], label=metric, linewidth=2)
plt.xlabel("Número de predictores (K)")
plt.ylabel("Valor de la métrica")
plt.title("Desempeño RF vs número de variables seleccionadas")
plt.legend()
plt.grid(True)
plt.show()
#%%
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# SELECCION K MEJORES = 14
selected = mrmr.mrmr_classif(X=X, y=y, K=26)
df_k = pd.concat([y, X[selected]], axis=1).dropna()
Xk, yk = df_k[selected], df_k["presence"]

X_train, X_test, y_train, y_test = train_test_split(
    Xk, yk, stratify=y, random_state=42, test_size=0.3
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Cross‐validation sobre sólo el set de entrenamiento
cv_results = cross_validate(
    rf,
    X_train, y_train,
    cv=5,
    scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
    return_train_score=False
)
y_pred = rf.predict(X_test)
print(f"Accuracy cross validation: {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}")

print(classification_report(y_test, y_pred))
# y_test y y_pred ya definidos
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
#%% obtenermos la importancia
import pandas as pd
import numpy as np

# 1) Construir un DataFrame con importancias
importances = pd.Series(rf.feature_importances_, index=X_train.columns)

# 2) Ordenar de mayor a menor
importances = importances.sort_values(ascending=False)

# 3) Mostrar las primeras filas
print("Importance RF")
print(importances.head(26))

# 4) (Opcional) Gráfico de barras de las 10 importancias más altas
import matplotlib.pyplot as plt

top10 = importances.head(26)
plt.figure(figsize=(8, 6))
plt.barh(top10.index[::-1], top10.values[::-1])  # invierto para que el más importante quede arriba
plt.xlabel("Importancia relativa")
plt.title("Top 10 predictores - Random Forest")
plt.tight_layout()
plt.show()
#%% 2) Guárdalo en un fichero 
import joblib

joblib.dump(rf, 'rf_model2.joblib')
#%% Cortamoss los archivos que me son necesarios
from shapely.geometry import box
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
import geopandas as gpd
import rasterio, os, glob

# 1) Carga tu GeoDataFrame y calcula la caja mínima
minx, miny, maxx, maxy = gdf_pres.total_bounds
bbox = box(minx, miny, maxx, maxy)    # polígono rectangular

# 2) Lee la capa de referencia para CRS/transform/resolution
fc_path = 'SoilMaps_MEAN/FC.30-60cm.tif'
with rasterio.open(fc_path) as fc:
    fc_crs       = fc.crs
    fc_transform = fc.transform
    fc_width     = fc.width
    fc_height    = fc.height
    fc_profile   = fc.profile.copy()

# 3) Procesa cada ráster
input_folder  = 'Bioclim_1959_2025'
output_folder = 'predictores'
os.makedirs(output_folder, exist_ok=True)


# --- 3) Construye la lista de archivos a procesar ---
all_tifs = glob.glob(os.path.join(input_folder, '*.tif'))
raster_files = [
    fp for fp in all_tifs
    if os.path.splitext(os.path.basename(fp))[0] in selected
]

print("Archivos seleccionados para procesar:")
for fp in raster_files:
    print("  -", os.path.basename(fp))

# --- 4) Procesa cada ráster seleccionado ---
for in_fp in raster_files:
    name = os.path.basename(in_fp)
    print(f"\nProcesando {name}…")
    
    # 4.1) Reproyecta / remuestrea al grid de la capa FC
    tmp_fp = os.path.join(output_folder, f"tmp_{name}")
    with rasterio.open(in_fp) as src, rasterio.open(tmp_fp, 'w', **fc_profile) as dst:
        for b in range(1, src.count + 1):
            reproject(
                source      = rasterio.band(src, b),
                destination = rasterio.band(dst, b),
                src_transform = src.transform,
                src_crs       = src.crs,
                dst_transform = fc_transform,
                dst_crs       = fc_crs,
                dst_width     = fc_width,
                dst_height    = fc_height,
                resampling    = Resampling.nearest
            )
    print("  → Reproyectado/remuestreado al grid de referencia")

    # 4.2) Recorta usando la caja (bbox)
    with rasterio.open(tmp_fp) as src2:
        out_img, out_transform = mask(
            src2,
            shapes=[bbox],
            crop=True,
            all_touched=False
        )
        out_meta = src2.meta.copy()
        out_meta.update({
            'driver':    'GTiff',
            'height':    out_img.shape[1],
            'width':     out_img.shape[2],
            'transform': out_transform,
            'crs':       fc_crs,
            'compress':  'lzw',
            'predictor': 2
        })

    # 4.3) Guarda el resultado final
    out_fp = os.path.join(output_folder, f"{name}")
    with rasterio.open(out_fp, 'w', **out_meta) as dst:
        dst.write(out_img)
    print(f"  → Guardado: {out_fp}")

    # 4.4) Limpieza del archivo intermedio
    os.remove(tmp_fp)

print("\n Procesamiento completo sin inflar el tamaño de los archivos.")
#%%
from shapely.geometry import box
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
import geopandas as gpd
import rasterio, os, glob

output_folder = 'predictores'
# 1) Carga tu GeoDataFrame y calcula la caja mínima
minx, miny, maxx, maxy = gdf_pres.total_bounds
bbox = box(minx, miny, maxx, maxy)    # polígono rectangular
input_folder =   'SoilMaps_MEAN'
fc_path = 'SoilMaps_MEAN/FC.30-60cm.tif'
with rasterio.open(fc_path) as fc:
    fc_profile   = fc.profile.copy()
    fc_crs       = fc.crs
    fc_transform = fc.transform
    fc_width     = fc.width
    fc_height    = fc.height

# --- 3) Construye la lista de archivos a procesar ---
all_tifs = glob.glob(os.path.join(input_folder, '*.tif'))
raster_files = [
    fp for fp in all_tifs
    if os.path.splitext(os.path.basename(fp))[0] in selected
]

print("Archivos seleccionados para procesar:")
for fp in raster_files:
    print("  -", os.path.basename(fp))

# --- 4) Procesa cada ráster seleccionado ---
for in_fp in raster_files:
    name = os.path.basename(in_fp)
    print(f"\nProcesando {name}…")
    
    # 4.1) Reproyecta / remuestrea al grid de la capa FC
    tmp_fp = os.path.join(output_folder, f"tmp_{name}")
    with rasterio.open(in_fp) as src, rasterio.open(tmp_fp, 'w', **fc_profile) as dst:
        for b in range(1, src.count + 1):
            reproject(
                source      = rasterio.band(src, b),
                destination = rasterio.band(dst, b),
                src_transform = src.transform,
                src_crs       = src.crs,
                dst_transform = fc_transform,
                dst_crs       = fc_crs,
                dst_width     = fc_width,
                dst_height    = fc_height,
                resampling    = Resampling.nearest
            )
    print("  → Reproyectado/remuestreado al grid de referencia")

    # 4.2) Recorta usando la caja (bbox)
    with rasterio.open(tmp_fp) as src2:
        out_img, out_transform = mask(
            src2,
            shapes=[bbox],
            crop=True,
            all_touched=False
        )
        out_meta = src2.meta.copy()
        out_meta.update({
            'driver':    'GTiff',
            'height':    out_img.shape[1],
            'width':     out_img.shape[2],
            'transform': out_transform,
            'crs':       fc_crs,
            'compress':  'lzw',
            'predictor': 2
        })

    # 4.3) Guarda el resultado final
    out_fp = os.path.join(output_folder, f"{name}")
    with rasterio.open(out_fp, 'w', **out_meta) as dst:
        dst.write(out_img)
    print(f"  → Guardado: {out_fp}")

    # 4.4) Limpieza del archivo intermedio
    os.remove(tmp_fp)

print("\n✅ Procesamiento completo sin inflar el tamaño de los archivos.")
#%%
import os
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from rasterio.enums import Resampling
from rasterio import windows

# --------------------------------
# 1) Parámetros
# --------------------------------
input_raster_path = r'C:\JORGE_HERRERA\PHD\curso_ing_ambiental_UC\SoilMaps_MEAN\coverage_mapbiomas.tif'
output_folder = r'C:\JORGE_HERRERA\PHD\curso_ing_ambiental_UC\predictores'
os.makedirs(output_folder, exist_ok=True)

clases_a_extraer = [3, 21,24,12,33,29,25]

# --------------------------------
# 2) Cargar GDF y calcular bounding box
# --------------------------------
#gdf_pres = gpd.read_file(r'C:\JORGE_HERRERA\PHD\curso_ing_ambiental_UC\gdf_pres.shp')  # si no está cargado
minx, miny, maxx, maxy = gdf_pres.total_bounds
bbox = box(minx, miny, maxx, maxy)

# --------------------------------
# 3) Cargar capa de referencia (FC) para definir proyección y resolución
# --------------------------------
fc_path = 'SoilMaps_MEAN/FC.30-60cm.tif'
with rasterio.open(fc_path) as fc:
    fc_crs       = fc.crs
    fc_transform = fc.transform
    fc_width     = fc.width
    fc_height    = fc.height
    fc_profile   = fc.profile.copy()

# --------------------------------
# 4) Leer el raster de entrada una sola vez
# --------------------------------
with rasterio.open(input_raster_path) as src:
    raster_data = src.read(1)
    src_transform = src.transform
    src_crs       = src.crs

# --------------------------------
# 5) Procesar cada clase
# --------------------------------
for clase in clases_a_extraer:
    print(f"\nProcesando clase {clase}...")

    # --- 5.1) Crear máscara binaria (0/1)
    binario = np.where(raster_data == clase, 1, 0).astype(np.uint8)

    # --- 5.2) Crear archivo temporal reproyectado al grid del FC
    tmp_path = os.path.join(output_folder, f"_tmp_mapbio__{clase}.tif")
    with rasterio.open(tmp_path, 'w', **fc_profile) as dst:
        reproject(
            source=binario,
            destination=rasterio.band(dst, 1),
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=fc_transform,
            dst_crs=fc_crs,
            dst_width=fc_width,
            dst_height=fc_height,
            resampling=Resampling.nearest
        )

    print("  → Reproyectado al grid de FC")

    # --- 5.3) Recortar con bbox
    with rasterio.open(tmp_path) as reproj:
        out_img, out_transform = mask(
            reproj,
            shapes=[bbox],
            crop=True,
            all_touched=False
        )
        out_meta = reproj.meta.copy()
        out_meta.update({
            'height': out_img.shape[1],
            'width': out_img.shape[2],
            'transform': out_transform,
            'compress': 'lzw',
            'predictor': 2,
            'nodata': 0
        })

    # --- 5.4) Guardar resultado final
    out_fp = os.path.join(output_folder, f"mapbio__{clase}.tif")
    with rasterio.open(out_fp, 'w', **out_meta) as dst:
        dst.write(out_img)

    print(f"  → Guardado: {out_fp}")

print("\n✅ Todos los mapas binarios han sido creados y recortados.")
#%%
import os
import numpy as np
import joblib
import rasterio
from rasterio.windows import Window
import rioxarray as riox
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
rf = joblib.load("rf_model2.joblib")
# —————————————————————————————————————————————
model = joblib.load("rf_model2.joblib")
variables = list(model.feature_names_in_)

carpeta_rasters = 'predictores'
archivos_rasters = [
    os.path.join(carpeta_rasters, f)
    for f in os.listdir(carpeta_rasters)
    if f.lower().endswith('.tif')
]
#_______________________________________________
# ordenamos raster
# mapea cada nombre de variable a su posición en la lista
orden = {var: i for i, var in enumerate(variables)}

# ordena los paths según el nombre base coincida con variables
archivos_rasters_ordenados = sorted(
    archivos_rasters,
    key=lambda path: orden.get(
        os.path.splitext(os.path.basename(path))[0],
        float('inf')      # los que no estén en variables van al final
    )
)

rasters = [
    riox.open_rasterio(path, chunks={'y':500,'x':500})
        .squeeze(drop=True)
        .rename(os.path.splitext(os.path.basename(path))[0])
    for path in archivos_rasters_ordenados
]

data_set = xr.merge(rasters)
print(data_set)  # comprueba variables y dimensiones



# —————————————————————————————————————————————
# 3) Prepara el perfil del GeoTIFF de salida
# —————————————————————————————————————————————
with rasterio.open(archivos_rasters[0]) as src0:
    profile = src0.profile.copy()

profile.update(
    count=1,                # una sola banda de salida
    dtype=rasterio.float32, # fuerza float32
    nodata=np.nan           # nodata = NaN
)

out_tif = "predicciones_proba_sinlc_rf.tif"
#%% # 4) Procesa bloque a bloque y escribe al vuelo
# —————————————————————————————————————————————
block_size = 4096  # ajusta según tu RAM

# dimensiones totales
n_y = data_set.dims['y']
n_x = data_set.dims['x']
#%%
with rasterio.open(out_tif, 'w', **profile) as dst:
    for y0 in range(0, n_y, block_size):
        for x0 in range(0, n_x, block_size):
            # tamaño del bloque (para los bordes)
            height = min(block_size, n_y - y0)
            width  = min(block_size, n_x - x0)

            # 4.1) extrae el subset del dataset
            subset = data_set.isel(
                y=slice(y0, y0 + height),
                x=slice(x0, x0 + width)
            )

            # 4.2) apila variables en un array NumPy (n_vars, h, w)
            arr = np.stack(
                [subset[var].values for var in variables],
                axis=0
            )

            # 4.3) aplana a (pixels, features) y predice
            n_vars, h, w = arr.shape
            flat = arr.reshape(n_vars, h*w).T            # (h*w, n_vars)
            mask = np.any(np.isnan(flat), axis=1)
            preds = np.full(h*w, np.nan, dtype=np.float32)
            if not mask.all():
                probas = model.predict_proba(flat[~mask])  # shape = (n_valid, 2)
                preds[~mask] = probas[:, 1]  
            pred_block = preds.reshape(h, w)

            # 4.4) escribe este bloque en el TIFF de salida
            window = Window(x0, y0, width, height)
            dst.write(pred_block, 1, window=window)

print("Predicciones guardadas en", out_tif)