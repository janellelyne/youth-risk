# Databricks notebook source
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder


#Using 4 of the 6 dataset sets from the Youth Risk Behavior Surveillance System, 
#we are going to perform exploratory analysis on the data.

#Loading the .csv files into dataframes
#[Note] Replace paths with local directory 
alch_drug_csv = pd.read_csv("/dbfs/FileStore/tables/AlcoholUse.csv")
phys_act_csv = pd.read_csv("/dbfs/FileStore/tables/PhysicalActivity.csv")
tobacco_use_csv = pd.read_csv("/dbfs/FileStore/tables/TabaccoUse.csv")
weight_ctrl_csv = pd.read_csv("/dbfs/FileStore/tables/WeightControl.csv")

#Restrict datasets to features we want to analyze. 
alch_drug = alch_drug_csv.filter(["Year", "State", "Location", "Abuse_catagory", "Abuse_type",
                                  "Sex", "Race", "Grade"], axis=1)
phys_act = phys_act_csv.filter(["Year", "State", "Location", "PE_attendance", "PE_attendance_type", 
                                "weekly_PE_attendance", "Sex", "Race", "Grade"], axis=1)
tobacco_use = tobacco_use_csv.filter(["Year", "State", "Location", "Topic", "Type_of_use", 
                                      "Usage_frequency", "Sex", "Race", "Grade"], axis=1)
weight_ctrl = weight_ctrl_csv.filter(["Year", "State", "Location", "Self_description", "Actual_description",
                                      "Sex", "Race", "11th"], axis=1)


#Remove columns with null values
phys_act = phys_act.dropna()
alch_drug = alch_drug.dropna()
tobacco_use = tobacco_use.dropna()
weight_ctrl = weight_ctrl.dropna()

#For the Grade and Sex features, lets restrict the data sets to Males/Females in 9th-12th grade
phys_act = phys_act[(phys_act.Grade != 'Total') | (phys_act.Sex != 'Total') |(phys_act.Race != 'Total')]
alch_drug = alch_drug[(alch_drug.Grade != 'Total') | (alch_drug.Sex != 'Total') |(alch_drug.Race != 'Total')]
tobacco_use = tobacco_use[(tobacco_use.Grade != 'Total')| (tobacco_use.Sex != 'Total') | (tobacco_use.Race != 'Total') ]
weight_ctrl.rename(columns={'11th': 'Grade'}, inplace=True)
weight_ctrl = weight_ctrl[(weight_ctrl.Grade != 'Total') | (weight_ctrl.Sex != 'Total') | (weight_ctrl.Race != 'Total')]

#Now lets choose a subset of features to have numbered aliases 
pa = phys_act[["Year", "State", "PE_attendance", "PE_attendance_type", "weekly_PE_attendance", "Sex", "Race", "Grade"]]
ad = alch_drug[["Year", "State", "Abuse_catagory", "Abuse_type", "Sex", "Race", "Grade"]]
tu = tobacco_use[["Year", "State", "Topic", "Type_of_use", "Usage_frequency", "Sex", "Race", "Grade"]]
wc = weight_ctrl[["Year", "State", "Self_description", "Actual_description", "Sex", "Race", "Grade"]]

#We will encode these catagorical features into numbers
#This will make it easier to perform clustering 
pa = pa.apply(LabelEncoder().fit_transform)
ad = ad.apply(LabelEncoder().fit_transform)
tu = tu.apply(LabelEncoder().fit_transform)
wc = wc.apply(LabelEncoder().fit_transform)



#Lets add some more feautres that might be useful.
#TODO



