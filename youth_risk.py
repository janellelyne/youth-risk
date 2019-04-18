from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 


#Using 4 of the 6 dataset sets from the Youth Risk Behavior Surveillance System, 
#we are going to perform exploratory analysis on the data.

#Loading the .csv files into dataframes
alch_drug = pd.read_csv("/dbfs/FileStore/tables/AlcoholUse.csv")
phys_act = pd.read_csv("/dbfs/FileStore/tables/PhysicalActivity.csv")
tabacco_use = pd.read_csv("/dbfs/FileStore/tables/TabaccoUse_use.csv")
weight_ctrl = pd.read_csv("/dbfs/FileStore/tables/Weight.csv")



