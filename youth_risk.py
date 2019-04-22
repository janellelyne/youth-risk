# Databricks notebook source
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

sns.set(style="darkgrid")
colors = ["amber", "windows blue", "greyish", "faded green", "dusty purple"]
sns.set_palette(sns.xkcd_palette(colors))
sns.set_context("notebook", 1.5)
alpha = 0.7


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

#Lets see where we are getting our data from
#Lets plot the locations of all the datasets
df_all_locations = alch_drug[['Location']].append([phys_act[['Location']], tobacco_use[['Location']], weight_ctrl[['Location']]])
m = df_all_locations['Location'].str.contains(',')
df_all_locations = df_all_locations[m]
figure = plt.figure(figsize=(10,10))
figure = figure.set_tight_layout({"pad": .0})
sns.countplot(y='Location', data=df_all_locations, alpha=alpha)
plt.title('Data by Location')
plt.axvline(x=50, color='k')
display(figure)


#Lets plot the data by year
df_all_years = alch_drug[['Year']].append([phys_act[['Year']], tobacco_use[['Year']], weight_ctrl[['Year']]])
figure = plt.figure(figsize=(10,10))
figure = figure.set_tight_layout({"pad": .0})
sns.countplot(x='Year', data=df_all_years, alpha=alpha)
plt.title('Data by Year')
plt.axhline(y=200, color='k')
display(figure)

#Lets plot the data by Gender and Grade
df_all_sexes = alch_drug[['Sex']].append([phys_act[['Sex']], tobacco_use[['Sex']], weight_ctrl[['Sex']]])
weight_ctrl.rename(columns={'11th': 'Grade'}, inplace=True)
df_all_grades = alch_drug[['Grade']].append([phys_act[['Grade']], tobacco_use[['Grade']], weight_ctrl[['Grade']]])
figure = plt.figure(figsize=(25,13))
# Sex
plt.subplot(221)
sns.countplot(x='Sex', data=df_all_sexes, alpha=alpha)
plt.title('Data by Gender', fontsize=20)
# Grade
plt.subplot(223)
sns.countplot(x='Grade', data=df_all_grades, alpha=alpha)
plt.title('Data by Grade', fontsize=20)


plt.tight_layout()
display(figure)
plt.show()


# Alcolhol Usage Frequency 
figure = plt.figure(figsize=(25,15))
figure = figure.set_tight_layout({"pad": .0})
plt.subplot(223)
sns.countplot(y='Abuse_type', data=alch_drug, alpha=alpha)
plt.title('Alcohol and Drug Abuse Type', fontsize=25)
plt.axhline(y=200, color='k')
display(figure)


# Physical Activity Report
figure = plt.figure(figsize=(25,15))
figure = figure.set_tight_layout({"pad": .0})
plt.subplot(223)
sns.countplot(y='PE_attendance_type', data=phys_act, alpha=alpha)
plt.title('Physical Activity Report', fontsize=25)
plt.axhline(y=200, color='k')
display(figure)


# Tobacco Usage Frequency
figure = plt.figure(figsize=(25,15))
figure = figure.set_tight_layout({"pad": .0})
plt.subplot(223)
sns.countplot(y='Usage_frequency', data=tobacco_use, alpha=alpha)
plt.title('Tobacco Usage Frequency', fontsize=25)
plt.axhline(y=200, color='k')
display(figure)


#DATA CLEANING
#Remove columns with null values
phys_act = phys_act.dropna()
alch_drug = alch_drug.dropna()
tobacco_use = tobacco_use.dropna()
weight_ctrl = weight_ctrl.dropna()

#For the Grade and Sex features, lets restrict the data sets to Males/Females in 9th-12th grade
phys_act = phys_act[(phys_act.Grade != 'Total') | (phys_act.Sex != 'Total') |(phys_act.Race != 'Total')]
alch_drug = alch_drug[(alch_drug.Grade != 'Total') | (alch_drug.Sex != 'Total') |(alch_drug.Race != 'Total')]
tobacco_use = tobacco_use[(tobacco_use.Grade != 'Total')| (tobacco_use.Sex != 'Total') | (tobacco_use.Race != 'Total') ]
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
