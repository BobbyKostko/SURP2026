from astropy.io import fits
from scipy.ndimage import median_filter
import numpy as np
from photutils.detection import DAOStarFinder
from scipy.optimize import curve_fit, linear_sum_assignment
import matplotlib.pyplot as plt
import math
import pandas as pd
import re
import matplotlib.colors as mcolors
import colorsys
import os

rows = []
output_csv = r"Dark_Current_Summary_Stats.csv"
filenames = [r"Y:\2D\20260406\iLocater_lab_20260406_0001_hxrgproc.fits",
r"Y:\2D\20260406\iLocater_lab_20260406_0002_hxrgproc.fits",
r"Y:\2D\20260407\iLocater_lab_20260407_0001_hxrgproc.fits",
r"Y:\2D\20260407\iLocater_lab_20260407_0002_hxrgproc.fits",
r"Y:\2D\20260409\iLocater_lab_20260409_0001_hxrgproc.fits",
r"Y:\2D\20260409\iLocater_lab_20260409_0002_hxrgproc.fits",
r"Y:\2D\20260410\iLocater_lab_20260410_0001_hxrgproc.fits",
r"Y:\2D\20260410\iLocater_lab_20260410_0002_hxrgproc.fits",
r"Y:\2D\20260411\iLocater_lab_20260411_0001_hxrgproc.fits",
r"Y:\2D\20260411\iLocater_lab_20260411_0002_hxrgproc.fits",
r"Y:\2D\20260412\iLocater_lab_20260412_0001_hxrgproc.fits",
r"Y:\2D\20260412\iLocater_lab_20260412_0002_hxrgproc.fits",
r"Y:\2D\20260413\iLocater_lab_20260413_0001_hxrgproc.fits",
r"Y:\2D\20260413\iLocater_lab_20260413_0002_hxrgproc.fits",
r"Y:\2D\20260414\iLocater_lab_20260414_0001_hxrgproc.fits",
r"Y:\2D\20260414\iLocater_lab_20260414_0002_hxrgproc.fits",
r"Y:\2D\20260415\iLocater_lab_20260415_0001_hxrgproc.fits",
r"Y:\2D\20260415\iLocater_lab_20260415_0002_hxrgproc.fits"]
temps = [130.738, 130.755, 121.429, 121.461, 102.087, 102.078, 95.8237, 95.8189, 90.8267, 90.8215, 88.1519, 88.1490, 86.4587, 86.4556, 84.8636, 84.8621, 83.7054, 83.7017]

def process_fits(filenames, temps = temps, hot_mask = None):
    for filename in filenames:
        print(filename)
        data = fits.getdata(filename)

        mean = np.mean(data)
        std = np.std(data)
        var = std**2

        #build a list of stats for the files in the input list
        rows.append({
            "Filename": os.path.basename(filename),
            "Mean": mean,
            "Standard Deviation": std,
            "Variance": var
        })
    
    #plot data vs temperature
    df = pd.DataFrame(rows)
    plt.figure()
    plt.scatter(temps, df["Mean"])
    plt.xlabel("Detector Temperature (K)")
    plt.ylabel("Mean Dark Current (Image-Wide)")
    plt.show

    #turn the data list into an exported csv
    
    df.to_csv(output_csv, index=False)

process_fits(filenames)

