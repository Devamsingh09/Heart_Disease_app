# Heart_Disease_app
I have created this project by my own , it will classify if the person has heart related disease or not and this project can be very benefitial for the medical institutes.
The model I used finally after GridSearcgCV was RandomForestClassifier with parameters "max_depth = 9, min_samples_leaf = 9".
Data has been collected from Kaggle Website and then I perfomed Data Cleaning, checked unique values of the columns. Plot the scatter charts for various variables. Then plotted bar charts, distribution chart. Then used Z_Score method to remove outliers, perfomed LabelEncoding and then after OneHotEncoding. Divied the datset into training and testing part with test set of 25%. Finally I perfomed GRidSearchCV. The best got accuracy was atround 88%.

