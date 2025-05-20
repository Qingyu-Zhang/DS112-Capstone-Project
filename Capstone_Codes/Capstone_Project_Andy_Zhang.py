# -*- coding: utf-8 -*-
"""
Created on Fri May  3 21:57:24 2024

@author: Qingyu Zhang / Andy
"""

#Capstone project DS112 2024
#Qingyu Zhang / Andy
#N-number: N19903322
#Email: qz2247@nyu.edu
#Spring 2024, Principles of Data Sience Pascal Wallisch



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from scipy.special import expit # this is the logistic sigmoid function


import random
mySeed = 19903322 # My N-number
np.random.seed(mySeed) # Initialize RNG with the my N-number as seed

alph = 0.05  #alpha threshold for significance test is set to 0.05


#A special indicator for data cleaning (whether to treat '0's in the "popularity" column as missing data or not. This part will be explained in the report):
dataCleaningMode = 2 #dataCleaningMode = 1 for not treatting '0's in "popularity" column as missing data; while dataCleaningMode = 2 for treating '0's in "popularity" column as missing data (use row-wise removal to delete the rows that the value in "popularity" is '0').
#Note: Although I suspect that '0's in "popularity" might be missing data, I don't do data pruning here before any of questions starts.
#Instead, I will decide to how to prune the dataset at the beginning of each questions,separately.
#This is because I need the whole dataset in some questions even if I treat '0's in "popularity" as missing data. For instance, when I want the distributions of the 10 features in Question 1. And the linear regression in Question 5 (since Q5 only involves "energy" and "loudness", and data in these two columns are just ok)




dataWhole = pd.read_csv("spotify52kData.csv")
print(dataWhole.info()) #Show information of this dataframe
has_nan = dataWhole.isna().any().any()  #Check whether there is any nan in the dataframe. Return true if there is at least one nan in the dataframe.
print('Does this dataframe have any NaN:', has_nan) #Turns out there is no NaNs in this dataframe. So we don't need to handle NaNs.

print("************************")






#%% Question1
#Plot 10 features and check whether they are normal distribution
print('Question 1:')
# Features to examine for normality
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness','acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Create a 2x5 grid for histograms
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()

for i, feature in enumerate(features):
    data = dataWhole[feature]
    axes[i].hist(data, bins=30)
    axes[i].set_title(feature)

plt.tight_layout()
plt.show()


print("************************")




#%% Question2
#Is there a relationship between song length and popularity of a song? If so, is the relationship positive or negative?
print("Question 2:")
#Get the parts we need
dataQ2 = dataWhole[['duration', 'popularity']]


#if we want to treat '0's in "popularity" as missing data and row-wise removal of corresponding rows:
if dataCleaningMode == 2:
    dataQ2 = dataQ2[dataQ2['popularity'] != 0]  #Row-wise removal


#Convert to numpy
dataQ2_np = dataQ2.to_numpy()

x_Q2 = dataQ2_np[:,0].reshape(len(dataQ2_np),1)
y_Q2 = dataQ2_np[:,1]

ModelQ2 = LinearRegression().fit(x_Q2, y_Q2)
y_Pred_Q2 = ModelQ2.predict(x_Q2)


#Calculate Pearson correlation
r_Q2 = np.corrcoef(dataQ2_np[:,0],y_Q2) 
print('Pearson correlation for Duration and Popularity:',np.round(r_Q2[0,1],3))

#plot
plt.plot(dataQ2_np[:,0],y_Q2,'o',markersize=3) 
plt.xlabel('Duration') 
plt.ylabel('Popularity')  
plt.plot(dataQ2_np[:,0],y_Pred_Q2,color='orange',linewidth=3)
plt.title('Relation between Song Length and Popularity, r = {:.3f}'.format(r_Q2[0,1])) 
plt.show()


print("************************")





#%% Question3
#Are explicitly rated songs more popular than songs that are not explicit?
print('Question 3:')


dataQ3 = dataWhole

#if we want to treat '0's in "popularity" as missing data and row-wise removal of corresponding rows:
if dataCleaningMode == 2:
    dataQ3 = dataQ3[dataQ3['popularity'] != 0]  #Row-wise removal



#First check how many true and false in the explicit column. These determine the corresponding length of the two groups we will have (explicit group and non-explicit group).
explicit_counts = dataQ3['explicit'].value_counts()
print('Explicit and non-explicit songs counts:')
print(explicit_counts)
print()


#Divide popularity into two groups by whether explicit or not
popularity_Explicit = dataQ3[dataQ3['explicit'] == True]['popularity']
popularity_NonExplicit = dataQ3[dataQ3['explicit'] == False]['popularity']


#Convert them to numpy arrays
popularity_Explicit_np = popularity_Explicit.to_numpy()
popularity_NonExplicit_np = popularity_NonExplicit.to_numpy()



#Plot the two groups and see their distribution
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

#Subplot 1: Explicit songs popularity distribution
axs[0].hist(popularity_Explicit_np, bins=50, color='blue')
axs[0].set_title('Popularity Distribution (explicit=True)')
axs[0].set_xlabel('Popularity')
axs[0].set_ylabel('Frequency')

#Subplot 2: Non-explicit songs popularity distribution
axs[1].hist(popularity_NonExplicit_np, bins=50, color='green')
axs[1].set_title('Popularity Distribution (explicit=False)')
axs[1].set_xlabel('Popularity')
axs[1].set_ylabel('Frequency')

plt.suptitle('Popularity Distributions for Explicit vs. Non-Explicit songs')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()




#As we can see, neither group's popularity values distribute normally. 
#Also, popularity measured in the dataset is not categorical. Popularity also doesn't reduce itself to means.
#Therefore, I choose to compare the medians of the two groups' popularity and use Mann-Whitney U test to test significance.

#Compute medians for two groups
explicit_Popularity_median = np.median(popularity_Explicit_np)
nonexplicit_Popularity_median = np.median(popularity_NonExplicit_np)
print('Median popularity of explicit group:', explicit_Popularity_median)
print('Median popularity of non-explicit group:', nonexplicit_Popularity_median)


#Do the Mann-Whitney U test
u_Q3,p_Q3 = stats.mannwhitneyu(popularity_Explicit_np, popularity_NonExplicit_np)
print('u is', u_Q3,'; p-value is', p_Q3)


#print conclusion:
if p_Q3 < alph:
    print('p-value is smaller than alpha.')
    print('We drop the null hypothesis that there is no difference in popularity between explicit songs and non-explicit songs.')
else:
    print('p-value is not smaller than alpha.')
    print('We failed to drop the null hypothesis that there is no difference in popularity between explicit songs and non-explicit songs.')



print("************************")









#%%Question4 
#Are songs in major key more popular than songs in minor key?
print('Question 4:')

dataQ4 = dataWhole

#if we want to treat '0's in "popularity" as missing data and row-wise removal of corresponding rows:
if dataCleaningMode == 2:
    dataQ4 = dataQ4[dataQ4['popularity'] != 0]  #Row-wise removal




#check the number of major key songs and minor key songs
mode_counts = dataQ4['mode'].value_counts()
print('Major key and minor key songs counts. (1 = song is in major; 0 = song is in minor)')
print(mode_counts)
print()


#Divide popularity into two groups by whether the song is in major key or minor key
popularity_Major = dataQ4[dataQ4['mode'] == 1]['popularity']
popularity_Minor = dataQ4[dataQ4['mode'] == 0]['popularity']


#Convert them to numpy arrays
popularity_Major_np = popularity_Major.to_numpy()
popularity_Minor_np = popularity_Minor.to_numpy()



#Plot the two groups and see their distribution
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

#Subplot 1: Major key songs popularity distribution
axs[0].hist(popularity_Major_np, bins=50, color='orange')
axs[0].set_title('Popularity Distribution (Major Key)(mode = 1)')
axs[0].set_xlabel('Popularity')
axs[0].set_ylabel('Frequency')

#Subplot 2: Minor key songs popularity distribution
axs[1].hist(popularity_Minor_np, bins=50, color='purple')
axs[1].set_title('Popularity Distribution (Minor Key)(mode = 0)')
axs[1].set_xlabel('Popularity')
axs[1].set_ylabel('Frequency')

plt.suptitle('Popularity Distributions for Major Key vs. Minor Key songs')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()




#As we can see, neither group's popularity values distribute normally. 
#Also, popularity measured in the dataset is not categorical. Popularity also doesn't reduce itself to means.
#Therefore, I choose to compare the medians of the two groups' popularity and use Mann-Whitney U test to test significance.

#Compute medians for two groups
major_Popularity_median = np.median(popularity_Major_np)
minor_Popularity_median = np.median(popularity_Minor_np)
print('Median popularity of Major Key group:', major_Popularity_median)
print('Median popularity of Minor Key group:', minor_Popularity_median)



#Do the Mann-Whitney U test
u_Q4,p_Q4 = stats.mannwhitneyu(popularity_Major_np, popularity_Minor_np)
print('u is', u_Q4,'; p-value is', p_Q4)



#print conclusion:
if p_Q4 < alph:
    print('p-value is smaller than alpha.')
    print('We drop the null hypothesis that there is no difference in popularity between major key songs and minor key songs.')
else:
    print('p-value is not smaller than alpha.')
    print('We failed to drop the null hypothesis that there is no difference in popularity between major key songs and minor key songs.')





print("************************")







#%% Question5
#Energy is believed to largely reflect the “loudness” of a song. Can you substantiate (or refute) that this is the case?
print('Question 5:')
dataQ5 = dataWhole[['energy', 'loudness']]

#Convert to numpy
dataQ5_np = dataQ5.to_numpy()

#Compute Pearson correlation between energy and loudness on the full set
r_Q5 = np.corrcoef(dataQ5_np[:,0],dataQ5_np[:,1]) 
print('Pearson correlation for Energy and Loudness:',np.round(r_Q5[0,1],3))

#Plot the scatter plot of Engery and Loudness
plt.plot(dataQ5_np[:,0], dataQ5_np[:,1] ,'o',markersize=3) 
plt.xlabel('Energy') 
plt.ylabel('Loudness')  
plt.title('Energy and Loudness Scatter Plot (Full Set), r = {:.3f}'.format(r_Q5[0,1])) 
plt.show()



#Set x and y
x_Q5 = dataQ5_np[:,0].reshape(len(dataQ5_np),1)
y_Q5 = dataQ5_np[:,1]


#Cross-validation
xTrain, xTest , yTrain, yTest = train_test_split(x_Q5, y_Q5, test_size=0.5) #Because of the N-number seed I gave to RNG, the random state is always my seed

#Use Train set for fitting the model only
ModelQ5 = LinearRegression().fit(xTrain, yTrain)

#Get predictions from xTest
y_Pred_Q5 = ModelQ5.predict(xTest)

#Use R-Square to evaluate the model. Compute the R-Square on the test set
rSq_Q5 = ModelQ5.score(xTest, yTest)

#Plot Linear Regression on test set with the model trained by train set
plt.plot(xTest, yTest ,'o',markersize=3) 
plt.xlabel('Energy') 
plt.ylabel('Loudness')  
plt.plot(xTest,y_Pred_Q5,color='orange',linewidth=3)
plt.title('Energy and Loudness Regression (Test Set), R^2 = {:.3f}'.format(rSq_Q5)) 
plt.show()

print('R-Square of the linear regression model for Energy and Loudness:',rSq_Q5)


print("************************")



#%% Question6
#Which of the 10 individual (single) song features from question 1 predicts popularity best? How good is this “best” model?
print('Question 6:')

dataQ6 = dataWhole

#if we want to treat '0's in "popularity" as missing data and row-wise removal of corresponding rows:
if dataCleaningMode == 2:
    dataQ6 = dataQ6[dataQ6['popularity'] != 0]  #Row-wise removal



results_Q6 = [] #Store the names, R-Square, and RMSE of each column

X_Q6 = dataQ6[['duration', 'danceability', 'energy', 'loudness', 'speechiness','acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
y_Q6 = dataQ6['popularity'].to_numpy()



#Use for loop to do 10 single-independent variable linear regression on each of the 10 features and record corresponding R-Square and RMSE
for column in X_Q6.columns:
    x_Q6 = X_Q6[[column]].to_numpy()
    xTrain, xTest , yTrain, yTest = train_test_split(x_Q6, y_Q6, test_size=0.5) #cross-validation
    
    #fit model with train set
    ModelQ6 = LinearRegression().fit(xTrain, yTrain)
    
    #get predictions on xTest
    y_Pred_Q6 = ModelQ6.predict(xTest)
    
    #compute R-Square
    rSq_Q6 = ModelQ6.score(xTest, yTest)
    
    #compute RMSE
    rmse = np.sqrt(np.mean((yTest - y_Pred_Q6) ** 2))
    
    #add result to the list
    results_Q6.append({'Column': column, 'R_squared': rSq_Q6, 'RMSE': rmse})


results_Q6_df = pd.DataFrame(results_Q6)
print(results_Q6_df)

max_r2_index = results_Q6_df['R_squared'].idxmax()  # Find the index with highest R-Square
max_r2_column = results_Q6_df.loc[max_r2_index, 'Column']  # Get the name of the column that has highest R-Square
max_r2_value = results_Q6_df.loc[max_r2_index, 'R_squared']  # Get the highest R-Square value
RMSE_value = results_Q6_df.loc[max_r2_index, 'RMSE']  # Get the corresponding RMSE value

print(f"Among the 10 features, the feature with the highest R-Square value is {max_r2_column} with an R-Square of {max_r2_value:.4f}, and a RMSE of {RMSE_value:.4f}")






print("************************")



#%% Question7
#Building a model that uses *all* of the song features from question 1, how well can you predict popularity now? How much (if at all) is this model improved compared to the best model in question 6). How do you account for this?
print('Question 7:')

dataQ7 = dataWhole[['popularity','duration', 'danceability', 'energy', 'loudness', 'speechiness','acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]


#if we want to treat '0's in "popularity" as missing data and row-wise removal of corresponding rows:
if dataCleaningMode == 2:
    dataQ7 = dataQ7[dataQ7['popularity'] != 0]  #Row-wise removal



#convert dataframe to array
dataQ7_np = dataQ7.to_numpy()

#set X (10 features) and y
X_Q7 = dataQ7_np[:,1:]
y_Q7 = dataQ7_np[:,0]


xTrain, xTest , yTrain, yTest = train_test_split(X_Q7, y_Q7, test_size=0.5) #cross-validation

#fit the model with train set
ModelQ7 = LinearRegression().fit(xTrain, yTrain)

#get predictions on test set
y_pred_Q7 = ModelQ7.predict(xTest)

#compute R-Square
rSq_Q7 = ModelQ7.score(xTest, yTest)


#plot prediction vs actual popularity
plt.plot(y_pred_Q7, yTest ,'o',markersize=3) 
plt.xlabel('Predictions of Popularity') 
plt.ylabel('Actual Popularity')  
plt.title('R^2 = {:.3f}'.format(rSq_Q7)) 
plt.show()

print('R-Square of this multiple linear regression model is:', rSq_Q7)



print("************************")





#%% Question8
#When considering the 10 song features above, how many meaningful principal components can you extract? What proportion of the variance do these principal components account for?
print('Question 8:')
#define a function for PCA
def pcaAnalysis(predictors,numPredictors):
    zscoredData = stats.zscore(predictors)
    pca = PCA().fit(zscoredData)
    eigVals = pca.explained_variance_
    loadings = pca.components_*-1
    origDataNewCoordinates = pca.fit_transform(zscoredData)*-1
    kaiserThreshold = 1
    numOfFactorsKaiser = np.count_nonzero(eigVals > kaiserThreshold)
    print('Number of factors selected by Kaiser criterion:', numOfFactorsKaiser)
    plt.bar(np.linspace(1,numPredictors,numPredictors),eigVals)
    plt.plot([0,numPredictors],[1,1],color='orange') # Orange Kaiser criterion line
    plt.title('Scree plot')
    plt.xlabel('Principal Components')
    plt.ylabel('Eigenvalues')
    plt.show()
    
    #Then show the loadings
    #plt.subplot(1,2,1) # Factor 1: 
    plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[0,:])
    plt.title('PC1')
    plt.show()
    #plt.subplot(1,2,2) # Factor 2:
    plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[1,:]) 
    plt.title('PC2')
    plt.show()
    #Factor 3:
    plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[2,:]*-1)
    plt.title('PC3')
    plt.show()
    
    varExplained = eigVals/sum(eigVals)*100
    print('Variance explained by each principal component:')
    for ii in range(len(varExplained)):
        print('Principal component',ii+1,': {:.3f}%'.format(varExplained[ii]))
        
    totalVarExplainedBySelectedFactors = 0
    for ii in range(numOfFactorsKaiser):
        totalVarExplainedBySelectedFactors = totalVarExplainedBySelectedFactors + varExplained[ii]
        
    print('Proportion of the variance accounted for by all the selected principal components: {:.3f}%'.format(totalVarExplainedBySelectedFactors) )
    
    return origDataNewCoordinates



#Get the 10 features we need
dataQ8 = dataWhole[['duration', 'danceability', 'energy', 'loudness', 'speechiness','acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
predictors_Q8 = dataQ8.to_numpy()


#To ascertain whether a PCA is indicated, let's look at the correlation heatmap
r_Q8 = np.corrcoef(predictors_Q8,rowvar=False)
plt.imshow(r_Q8) 
plt.colorbar()
plt.show()


#Do PCA and get rotated dataset
rotatedPredictors = pcaAnalysis(predictors_Q8, predictors_Q8.shape[1])
print(rotatedPredictors.shape)







print("************************")














#%% Question9
#Can you predict whether a song is in major or minor key from valence? If so, how good is this prediction? If not, is there a better predictor?
print('Question 9:')


def logistic_Regression(data):
    data_np = data.to_numpy()
    x = data_np[:,1].reshape(len(data_np),1) 
    y = data_np[:,0].astype(int)  
    
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    
    model = LogisticRegression().fit(x_train, y_train)
    
    y_prob = model.predict_proba(x_test)[:, 1]  
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    #Get name of the predictor and the name of the outcome
    x_name = data.columns[1]
    y_name = data.columns[0]

    #plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC: {x_name} predict {y_name}")
    plt.legend(loc="lower right")
    plt.show()
    
    
    #Get minimum and maximum of the array
    min_value = x_test.min()
    max_value = x_test.max()
    
    x1 = np.linspace(min_value, max_value,500)
    y1 = x1 * model.coef_ + model.intercept_
    sigmoid = expit(y1)
    plt.plot(x1,sigmoid.ravel(),color='red',linewidth=3) # the ravel function returns a flattened array
    plt.scatter(x_test,y_test,color='black')
    plt.hlines(0.5, min_value, max_value, colors='gray', linestyles='dotted')
    plt.xlabel(x_name)
    plt.xlim([min_value, max_value])
    plt.ylabel(y_name)
    plt.yticks(np.array([0,1]))
    plt.show()
    return model




#predict mode from valence
dataQ9_valence = dataWhole[['mode', 'valence']]  
logistic_Q9_valence = logistic_Regression(dataQ9_valence)


#I tried for other 9 features and PC1, turns out that speechiness and acousticness are both the 'best' predictor for 'mode' among them

#predict mode from PC1:
dataQ9_PC1 = dataWhole[['mode']]
dataQ9_PC1['PC1'] = rotatedPredictors[:,0]
logistic_Q9_modePC1 = logistic_Regression(dataQ9_PC1)


#predict mode from speechiness:
dataQ9_speechiness = dataWhole[['mode', 'speechiness']]
logistic_Q9_speechiness = logistic_Regression(dataQ9_speechiness)

#predict mode from acousticness
dataQ9_acousticness = dataWhole[['mode', 'acousticness']]
logistic_Q9_acousticness = logistic_Regression(dataQ9_acousticness)


print("************************")




#%% Question10
#Which is a better predictor of whether a song is classical music – duration or the principal components you extracted in question 8?
print("Question 10:")

dataQ10 = dataWhole.copy()

#convert values in track_genre column to 1 and 0.
dataQ10['track_genre'] = (dataQ10['track_genre'] == 'classical').astype(int)


#predict classical or not from duration:
dataQ10_duration = dataQ10[['track_genre', 'duration']]
logistic_Q10_duration = logistic_Regression(dataQ10_duration)


#predict classical or not from Principal component 1:
dataQ10_GenrePC1 = dataQ10[['track_genre']]
dataQ10_GenrePC1['PC1'] = rotatedPredictors[:,0]
logistic_GenrePC1 = logistic_Regression(dataQ10_GenrePC1)







#predict classical or not from the 3 chosen principal components:
x_PCs = rotatedPredictors[:,0:3]
y_Genre = dataQ10['track_genre'].to_numpy()


x_train, x_test, y_train, y_test = train_test_split(x_PCs, y_Genre, test_size=0.3)

model = LogisticRegression().fit(x_train, y_train)

y_prob = model.predict_proba(x_test)[:, 1]  
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)


#plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC: 3 PCs predict track_genre")
plt.legend(loc="lower right")
plt.show()



print("************************")







#%% Q11 测试随机森林
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


if dataCleaningMode == 1:
    dataQ11 = dataWhole[dataWhole['popularity'] != 0]  #Row-wise removal
    
else:
    dataQ11 = dataWhole

features = dataQ11[['duration', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
target = dataQ11['popularity']

# 标准化特征（可选，但通常对于随机森林来说不是必需的）
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)

# 构建随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # 可以调整n_estimators和其他参数

# 训练模型
rf_model.fit(X_train, y_train)

# 在测试集上进行预测
predictions = rf_model.predict(X_test)

# 使用R²评估模型
r2 = r2_score(y_test, predictions)

# 打印R²得分
print(f'R² score: {r2}')