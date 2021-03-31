import os, io
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
from PIL import Image

# CSV reader function

def csvReader(url):
    df = pd.read_csv(url, 
                    delimiter = ',', header = 0, na_filter=True, encoding="utf-8")

    df['date'] = pd.to_datetime(df['date'])

    # filter date range
    start_date = df['date'] > "2008-01-01"
    end_date = df['date'] < "2011-01-01"

    df = df[ start_date & end_date ]

    
    # checks for the value in the file column and multiply by 10e-6
    if 'value' in df:
        df = df[df['value'] != -32000] # filter outliers = -32000
        df['value'] = df['value'] * (10**-6)


    return df.set_index('date')


#============ READ EACH DATA CSV

# BAFOULABE
bafoulabe_1 = csvReader('data/series_13.81386_-10.82952_2007-01-01_2010-12-31_Bafoulabe.csv')
bafoulabe_10 = csvReader('data/JRC_2002_2020_-10.793_13.84_Bafoulabe.csv')

# KAYES
kayes_1 = csvReader('data/series_14.44858_-11.42722_2008-01-01_2010-12-31_Kayes.csv')
kayes_10 = csvReader('data/JRC_2002_2020_-11.455_14.524_Kayes.csv')

# BAKEL
bakel_1 = csvReader('data/series_14.89412_-12.45496_2007-01-01_2010-12-31_Bakel.csv')
bakel_10 = csvReader('data/JRC_2002_2020_-12.46_14.89_Bakel.csv')


#============ JOINT PLOT FOR EACH LOCATION

# BAFOULABE Plot 1x1 and 10x10 time series together
fig, ax = plt.subplots(figsize=(24, 12))
ax.plot(bafoulabe_1['WI-AMSRE-KA-H-DESC-K90_V003_1000_AVG'], linestyle='-', linewidth=1, label='1m resolution')
ax.plot(bafoulabe_10['value'], linestyle='-', linewidth=1, label='10m resolution')
ax.set_xlabel('Date')
ax.set_ylabel('values')
ax.set_title('BAFOULABE')
ax.legend()


png_bafoulabe = io.BytesIO()
fig.savefig(png_bafoulabe, format="png")

png_save = Image.open(png_bafoulabe)


png_save.save("output/bafoulabe.tiff")
png_bafoulabe.close()


# KAYES Plot 1x1 and 10x10 time series together
fig, ax = plt.subplots(figsize=(24, 12))
ax.plot(kayes_1['WI-AMSRE-KA-H-DESC-K90_V003_1000_AVG'], linestyle='-', linewidth=1, label='1m resolution')
ax.plot(kayes_10['value'], linestyle='-', linewidth=1, label='10m resolution')
ax.set_xlabel('Date')
ax.set_ylabel('values')
ax.set_title('KAYES')
ax.legend()


png_kayes = io.BytesIO()
fig.savefig(png_kayes, format="png")

png_save = Image.open(png_kayes)


png_save.save("output/kayes.tiff")
png_kayes.close()


# BAKEL Plot 1x1 and 10x10 time series together
fig, ax = plt.subplots(figsize=(24, 12))
ax.plot(bakel_1['WI-AMSRE-KA-H-DESC-K90_V003_1000_AVG'], linestyle='-', linewidth=1, label='1m resolution')
ax.plot(bakel_10['value'], linestyle='-', linewidth=1, label='10m resolution')
ax.set_xlabel('Date')
ax.set_ylabel('values')
ax.set_title('BAKEL')
ax.legend()

# Save the image in memory in PNG format
png_bakel = io.BytesIO()
fig.savefig(png_bakel, format="png")
# Load this image into PIL
png_save = Image.open(png_bakel)

# Save as TIFF
png_save.save("output/bakel.tiff")
png_bakel.close()

""" 
*** STOPPED USING PYTHON SCRIPT AND SWITCHED TO NOTEBOOK SINCE COLLAB PRIVATE GITHUB REPO ISSUES
 """
