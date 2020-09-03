Project Scope:

 The Center for Disease Control and Prevention (CDC) conducts an annual survey called the Youth Risk Behavior Surveillance System (YRBSS), which monitors priority health behaviors and experiences among high school students across the country. The Youth Risk Behavior Survey (YRBS) attempts to quantify the leading factors of HIV, drug use, and poor nutritional diets in young adults (CDC). YRBSS monitors six categories of health-related behaviors that contribute to the leading causes of death and disability among youth and adults.

Because of the nature of this data, we ultimately want to attempt our own exploration into the factors that cause alcohol abuse, physical inactivity, and tobacco use through our own methods of data visualization, classification, and cluster analysis. The 4 YRBSS datasets used in this project are provided by Kaggle.com. For the datasets, we chose to use Decision Tree Classifier, Decision Tree using Bagging Method, K-Means Clustering, and Agglomerative Clustering as our classification and clustering methods. These methods were chosen because they provide a range of different techniques, and by using all of them we hope to amalgamate a cohesive and complete solution.

We were able to classify alcohol and drug use based on our attributes with accuracies of nearly 85%. This indicates a high success rate, although our AUC of ROC only approx. 60%. Nevertheless, our 10-fold cross validation provided us with positive results, and shows good indication of a successful classification. We were less successful with clustering due to the nature of the data. Our clusters were unable to separate the data in a meaningful way, even though we tried both K-Means and Agglomerative clustering.


How to Run:

python youth_risk.python


Dependencies:
AlcoholUse.csv
PhysicalActivity.csv
TabaccoUse.csv
WeightControl.csv

Libraries:
sklearn
numpy
pandas
matplotlib
seaborn
