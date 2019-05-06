# Databricks notebook source
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import tree
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
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
alch_drug_csv = pd.read_csv("AlcoholUse.csv")
phys_act_csv = pd.read_csv("PhysicalActivity.csv")
tobacco_use_csv = pd.read_csv("TabaccoUse.csv")
weight_ctrl_csv = pd.read_csv("WeightControl.csv")


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

payg = pa[["Year", "PE_attendance"]]
kmeans = KMeans(n_clusters=4)
kmeans.fit(payg)
print(kmeans.cluster_centers_)
labels = kmeans.predict(payg)
kmeans = pd.DataFrame(labels)
payg.insert((payg.shape[1]),'kmeans',kmeans)

figure = plt.figure()
ax = figure.add_subplot(111)
scatter = ax.scatter(payg['Year'],payg['PE_attendance'], c=kmeans[0])
ax.set_xlabel('Year')
ax.set_ylabel('Physical_activity')
plt.colorbar(scatter)
display(figure)

adyg = ad[["Year", "Abuse_type"]]
kmeans = KMeans(n_clusters=4)
kmeans.fit(adyg)
print(kmeans.cluster_centers_)
labels = kmeans.predict(adyg)
kmeans = pd.DataFrame(labels)
adyg.insert((adyg.shape[1]),'kmeans',kmeans)

#Plot the clusters obtained using k means
figure = plt.figure()
ax = figure.add_subplot(111)
scatter = ax.scatter(adyg['Year'],adyg['Abuse_type'], c=kmeans[0])
ax.set_xlabel('Year')
ax.set_ylabel('Abuse_type')
plt.colorbar(scatter)
display(figure)


tuyg = tu[["Year", "Usage_frequency"]]
kmeans = KMeans(n_clusters=4)
kmeans.fit(tuyg)
print(kmeans.cluster_centers_)
labels = kmeans.predict(tuyg)
kmeans = pd.DataFrame(labels)
tuyg.insert((tuyg.shape[1]),'kmeans',kmeans)

#Plot the clusters obtained using k means
figure = plt.figure()
ax = figure.add_subplot(111)
scatter = ax.scatter(tuyg['Year'],tuyg['Usage_frequency'], c=kmeans[0])
ax.set_xlabel('Year')
ax.set_ylabel('Usage_frequency')
plt.colorbar(scatter)
display(figure)


wcyg = wc[["Year", "State"]]
kmeans = KMeans(n_clusters=4)
kmeans.fit(wcyg)
print(kmeans.cluster_centers_)
labels = kmeans.predict(wcyg)
kmeans = pd.DataFrame(labels)
wcyg.insert((wcyg.shape[1]),'kmeans',kmeans)

#Plot the clusters obtained using k means
figure = plt.figure()
ax = figure.add_subplot(111)
scatter = ax.scatter(wcyg['Year'],wcyg['State'], c=kmeans[0])
ax.set_xlabel('Year')
ax.set_ylabel('State')
plt.colorbar(scatter)
display(figure)

payg = pa[["Year", "PE_attendance"]]
paygsample = payg.sample(100)
agglo = AgglomerativeClustering(n_clusters=4)
labels = agglo.fit_predict(paygsample)
agglo = pd.DataFrame(labels)
paygsample.insert((paygsample.shape[1]),'agglo',agglo)

#Plot the clusters obtained using agglo
figure = plt.figure()
ax = figure.add_subplot(111)
scatter = ax.scatter(paygsample['Year'],paygsample['PE_attendance'], c=agglo[0])
ax.set_xlabel('Year')
ax.set_ylabel('Physical_activity')
plt.colorbar(scatter)
display(figure)

adyg = ad[["Year", "Abuse_type"]]
adygsample = adyg.sample(100)
agglo = AgglomerativeClustering(n_clusters=4)
labels = agglo.fit_predict(adygsample)
agglo = pd.DataFrame(labels)
adygsample.insert((adygsample.shape[1]),'agglo',agglo)

#Plot the clusters obtained using agglo
figure = plt.figure()
ax = figure.add_subplot(111)
scatter = ax.scatter(adygsample['Year'],adygsample['Abuse_type'], c=agglo[0])
ax.set_xlabel('Year')
ax.set_ylabel('Abuse_type')
plt.colorbar(scatter)
display(figure)

tuyg = tu[["Year", "Usage_frequency"]]
tuygsample = tuyg.sample(100)
agglo = AgglomerativeClustering(n_clusters=4)
labels = agglo.fit_predict(tuygsample)
agglo = pd.DataFrame(labels)
tuygsample.insert((tuygsample.shape[1]),'agglo',agglo)

#Plot the clusters obtained using agglo
figure = plt.figure()
ax = figure.add_subplot(111)
scatter = ax.scatter(tuygsample['Year'],tuygsample['Usage_frequency'], c=agglo[0])
ax.set_xlabel('Year')
ax.set_ylabel('Usage_frequency')
plt.colorbar(scatter)
display(figure)

wcyg = wc[["Year", "State"]]
wcygsample = wcyg.sample(100)
agglo = AgglomerativeClustering(n_clusters=4)
agglo.fit(wcygsample)
labels = agglo.fit_predict(wcygsample)
agglo = pd.DataFrame(labels)
wcygsample.insert((wcygsample.shape[1]),'agglo',agglo)

#Plot the clusters obtained using agglo
figure = plt.figure()
ax = figure.add_subplot(111)
scatter = ax.scatter(wcygsample['Year'],wcygsample['State'], c=agglo[0])
ax.set_xlabel('Year')
ax.set_ylabel('State')
plt.colorbar(scatter)
display(figure)

#Brute Force Trees

#Predicting the PE_attendance type
pas = pa.sample(150000)
pat = pa.drop(pas.index)
paty = pat[["PE_attendance"]]
pat = pat[["Sex", "Year", "Race", "Grade"]]
clf = tree.DecisionTreeClassifier(criterion = "entropy")
clf.fit(pas[["Sex", "Year", "Race", "Grade"]],pas[["PE_attendance"]])
returnAcc = clf.predict(pat)
accuracy = (np.sum((returnAcc > 5) == (paty.values.flatten() > 5))/len(paty))*100
print(accuracy)

#Predicting type of tobacco use
tus = tu.sample(150000)
tut = tu.drop(tus.index)
tuty = tut[["Type_of_use"]]
tut = tut[["Year", "State", "Topic","Sex", "Race"]]
clf = tree.DecisionTreeClassifier(criterion = "entropy")
clf.fit(tus[["Year", "State", "Topic","Sex", "Race"]],tus[["Type_of_use"]])
returnAcc = clf.predict(tut)
accuracy = (np.sum(returnAcc == tuty.values.flatten())/len(tuty))*100
print(accuracy)

#Predicting gender based of weight characteristics
wcs = wc.sample(150000)
wct = wc.drop(wcs.index)
wcty = wct[["Sex"]]
wct = wct[["Year", "State", "Actual_description", "Race"]]
clf = tree.DecisionTreeClassifier(criterion = "entropy")
clf.fit(wcs[["Year", "State", "Actual_description", "Race"]],wcs[["Sex"]])
returnAcc = clf.predict(wct)
accuracy = (np.sum((returnAcc == 1) == (wcty.values.flatten()==1))/len(wcty))*100
print(accuracy)

#Alcohol and Drug Abuse Category Decision Tree
#Predict whether student is more succeptable to 
#drug or alcohol use based on other attributes
#Brute force random sampling with Entropy Hueristic
ads = ad.sample(150000)
adt = ad.drop(ads.index)
adty = adt[["Abuse_catagory"]]
adt = adt[["Sex", "Year", "Race", "Grade"]]
adt_train, adt_test, adty_train, adty_test = train_test_split(adt, adty, random_state=1)
clfe = tree.DecisionTreeClassifier(criterion = "entropy")
clfe.fit(adt_train, adty_train)
adty_predict = clfe.predict(adt_test)
accuracy = accuracy_score(adty_test, adty_predict)
print(accuracy)

#The negative Alcohol labels are more than 2x less than 
#the drug use labels
#We need a balanced sample
ad_count = ad.groupby('Abuse_catagory')
ad_count.describe()

#Method 1 of handling unbalanced data
#Take random sampling of majority class so the two class counts are even
majority = ad[ad.Abuse_catagory==0]
minority = ad[ad.Abuse_catagory==1]
majority_downsampled = resample(majority, n_samples=716153, random_state=123)
downsampled = pd.concat([majority_downsampled, minority])
downsampled.groupby('Abuse_catagory').describe()

#Rerun Decision Tree with balanced data from downsampling
ads = downsampled.sample(1000000)
adt = ad.drop(ads.index)
adty = adt[["Abuse_catagory"]]
adt = adt[["Sex", "Year", "Race", "Grade"]]
adt_train, adt_test, adty_train, adty_test = train_test_split(adt, adty, random_state=1)
clfe2 = tree.DecisionTreeClassifier(criterion = "entropy")
clfe2.fit(adt_train, adty_train)
adty_predict = clfe2.predict(adt_test)
accuracy = accuracy_score(adty_test, adty_predict)
print(accuracy)

#Method 2 of handling unbalanced data
#Use the AUROC performance metric instead of Accuracy
# On brute force tree
prob_y_2 = clfe.predict_proba(adt_train)
prob_y_2 = [p[1] for p in prob_y_2]
print(roc_auc_score(adty_train, prob_y_2))

#Use the AUROC performance metric instead of Accuracy
#on the balanced data
prob_y_2 = clfe2.predict_proba(adt_train)
prob_y_2 = [p[1] for p in prob_y_2]
print(roc_auc_score(adty_train, prob_y_2))

# Lets perform the same process using:
#Brute force random sampling with Gini Impurity Hueristic
ads = ad.sample(150000)
adt = ad.drop(ads.index)
adty = adt[["Abuse_catagory"]]
adt = adt[["Sex", "Year", "Race", "Grade"]]
adt_train, adt_test, adty_train, adty_test = train_test_split(adt, adty, random_state=1)
clfg = tree.DecisionTreeClassifier(criterion = "gini")
clfg.fit(adt_train, adty_train)
adty_predict = clfg.predict(adt_test)
accuracy = accuracy_score(adty_test, adty_predict)
print(accuracy)

#Method 1 of handling unbalanced data
#Take random sampling of majority class so the two class counts are even
majority = ad[ad.Abuse_catagory==0]
minority = ad[ad.Abuse_catagory==1]
 
majority_downsampled = resample(majority, n_samples=716153, random_state=123)
downsampled = pd.concat([majority_downsampled, minority])
downsampled.groupby('Abuse_catagory').describe()
#Rerun Decision Tree with balanced data from downsampling
ads = downsampled.sample(1000000)
adt = ad.drop(ads.index)
adty = adt[["Abuse_catagory"]]
adt = adt[["Sex", "Year", "Race", "Grade"]]
adt_train, adt_test, adty_train, adty_test = train_test_split(adt, adty, random_state=1)
clfg2 = tree.DecisionTreeClassifier(criterion = "gini")
clfg2.fit(adt_train, adty_train)
adty_predict = clfg2.predict(adt_test)
accuracy = accuracy_score(adty_test, adty_predict)
print(accuracy)

#Method 2 of handling unbalanced data
#Use the AUROC performance metric instead of Accuracy
# On brute force tree
prob_y_2 = clfg.predict_proba(adt_train)
prob_y_2 = [p[1] for p in prob_y_2]
print(roc_auc_score(adty_train, prob_y_2))

#Use the AUROC performance metric instead of Accuracy
#on the balanced data
prob_y_2 = clfg2.predict_proba(adt_train)
prob_y_2 = [p[1] for p in prob_y_2]
print(roc_auc_score(adty_train, prob_y_2))

seed = 7
#Now we are going to use the Bagging method with our decision tree classifiers
#For 100 trees created from our dataset
#Balanced data and entropy hueristic
kfold = model_selection.KFold(n_splits=5)
num_trees = 100
model = BaggingClassifier(base_estimator=clfe2, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, adt_train, adty_train, cv=kfold)
print("Max: " + str(results.max()) + " Mean: " + str(results.mean()))

# COMMAND ----------

#Now we are going to use the Bagging method with our decision tree classifiers
#For 100 trees created from our dataset
#Balanced data and gini impurity heuristic
kfold = model_selection.KFold(n_splits=5)
num_trees = 100
model = BaggingClassifier(base_estimator=clfg2, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, adt_train, adty_train, cv=kfold)
print(results.max())
print("Max: " + str(results.max()) + " Mean: " + str(results.mean()))

#Now we are going to use the Bagging method with our decision tree classifiers
#For 100 trees created from our dataset
#Balanced data and entropy hueristic
kfold = model_selection.KFold(n_splits=10)
num_trees = 100
model = BaggingClassifier(base_estimator=clfe2, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, adt_train, adty_train, cv=kfold)
print("Max: " + str(results.max()) + " Mean: " + str(results.mean()))

# COMMAND ----------

#Now we are going to use the Bagging method with our decision tree classifiers
#For 100 trees created from our dataset
#Balanced data and gini impurity heuristic
kfold = model_selection.KFold(n_splits=10)
num_trees = 100
model = BaggingClassifier(base_estimator=clfg2, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, adt_train, adty_train, cv=kfold)
print(results.max())
print("Max: " + str(results.max()) + " Mean: " + str(results.mean()))


