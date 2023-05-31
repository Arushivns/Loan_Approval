#Load libraries

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import normaltest
import statsmodels.formula.api as smf
#plt.rcParams['figure.figsize']=(10,10)
sns.set()
sns.set(style="darkgrid")
import pylab as py

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge

from sklearn.compose import TransformedTargetRegressor

#NN
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

from functools import partial
from sklearn.metrics import accuracy_score
#from skopt import space
#from skopt import gp_minimize
import numpy as np

#%%

# custom function for easy and efficient analysis of numerical univariate
def UVA_numeric(data, var_group):
  ''' 
  Univariate_Analysis_numeric
  takes a group of variables (INTEGER and FLOAT) and plot/print all the descriptives and properties along with KDE.

  Runs a loop: calculate all the descriptives of i(th) variable and plot/print it
  '''

  size = len(var_group)
  plt.figure(figsize = (4*size,3), dpi = 100)
  
  #looping for each variable
  for j,i in enumerate(var_group):
    
    # calculating descriptives of variable
    mini = data[i].min()
    maxi = data[i].max()
    ran = data[i].max()-data[i].min()
    mean = data[i].mean()
    median = data[i].median()
    st_dev = data[i].std()
    skew = data[i].skew()
    kurt = data[i].kurtosis()

    # calculating points of standard deviation
    points = mean-st_dev, mean+st_dev

    #Plotting the variable with every information
    plt.subplot(1,size,j+1)
    sns.kdeplot(data[i], shade=True)
    sns.lineplot(points, [0,0], color = 'black', label = "std_dev")
    sns.scatterplot([mini,maxi], [0,0], color = 'orange', label = "min/max")
    sns.scatterplot([mean], [0], color = 'red', label = "mean")
    sns.scatterplot([median], [0], color = 'blue', label = "median")
    plt.xlabel('{}'.format(i), fontsize = 20)
    plt.ylabel('density')
    plt.title('std_dev = {}; kurtosis = {};\nskew = {}; range = {}\nmean = {}; median = {}'.format((round(points[0],2),round(points[1],2)),
                                                                                                   round(kurt,2),
                                                                                                   round(skew,2),
                                                                                                   (round(mini,2),round(maxi,2),round(ran,2)),
                                                                                                   round(mean,2),
                                                                                                   round(median,2)))


#%%
# Normality Check function

def normality_check(data,var_group):
    '''
    Normality Check function
    '''
    size = len(var_group)
    #plt.figure(figsize = (5*size,3), dpi = 100)
    # qqplot
    for j,i in enumerate(var_group):
        skew=pd.DataFrame.skew(data[i])
        print("Q_Q plot for ", i)
        sm.qqplot(data[i],line='s')
        py.show()   


#%%
# EDA

def UVA_outlier(data, var_group, include_outlier = True):
  '''
  Univariate_Analysis_outlier:
  takes a group of variables (INTEGER and FLOAT) and plot/print boxplot and descriptives\n
  Runs a loop: calculate all the descriptives of i(th) variable and plot/print it \n\n

  data : dataframe from which to plot from\n
  var_group : {list} type Group of Continuous variables\n
  include_outlier : {bool} whether to include outliers or not, default = True\n
  '''

  size = len(var_group)
  plt.figure(figsize = (5*size,4), dpi = 100)
  
  #looping for each variable
  for j,i in enumerate(var_group):
    
    # calculating descriptives of variable
    quant25 = data[i].quantile(0.25)
    quant75 = data[i].quantile(0.75)
    IQR = quant75 - quant25
    med = data[i].median()
    whis_low = med-(1.5*IQR)
    whis_high = med+(1.5*IQR)

    # Calculating Number of Outliers
    outlier_high = len(data[i][data[i]>whis_high])
    outlier_low = len(data[i][data[i]<whis_low])

    if include_outlier == True:
      print(include_outlier)
      #Plotting the variable with every information
      plt.subplot(1,size,j+1)
      sns.boxplot(data[i], orient="v")
      plt.ylabel('{}'.format(i))
      plt.title('With Outliers\nIQR = {}; Median = {} \n 2nd,3rd  quartile = {};\n Outlier (low/high) = {} \n'.format(
                                                                                                   round(IQR,2),
                                                                                                   round(med,2),
                                                                                                   (round(quant25,2),round(quant75,2)),
                                                                                                   (outlier_low,outlier_high)
                                                                                                   ))
      
    else:
      # replacing outliers with max/min whisker
      data2 = data[var_group][:]
      data2[i][data2[i]>whis_high] = whis_high+1
      data2[i][data2[i]<whis_low] = whis_low-1
      
      # plotting without outliers
      plt.subplot(1,size,j+1)
      sns.boxplot(data2[i], orient="v")
      plt.ylabel('{}'.format(i))
      plt.title('Without Outliers\nIQR = {}; Median = {} \n 2nd,3rd  quartile = {};\n Outlier (low/high) = {} \n'.format(
                                                                                                   round(IQR,2),
                                                                                                   round(med,2),
                                                                                                   (round(quant25,2),round(quant75,2)),
                                                                                                   (outlier_low,outlier_high)
                                                                                                   ))

def Outliers_Distribution (df):
    '''
    Parameters
    ----------
    df : dataframe (with target variable)

    Returns
    -------
    df_outliers: dataframe with outliers according to IQR
    distribution: how are the outliers distributed according to the target variable

    '''
    df_outliers = pd.DataFrame()
    for var in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']:
        q1 = np.percentile(df[var], 25)
        q3 = np.percentile(df[var], 75)
        iqr = q3 - q1
        # calculate the lower and upper bounds for outliers
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        print(var, ": ", lower_bound, ", ", upper_bound)
        #store outliers in subset
        aux = df[(df[var]<lower_bound) | (df[var]>upper_bound)] 
        df_outliers = df_outliers.append(aux)
    df_outliers = df_outliers.drop_duplicates()
    
    df_outliers.groupby("Loan_Status").size()
    distribution = round(df_outliers.Loan_Status.value_counts(normalize = True)*100,2)
    
    return distribution, df_outliers


#%%

def UVA_category(data, var_group):

  '''
  Univariate_Analysis_categorical
  takes a group of variables (category) and plot/print all the value_counts and barplot.
  '''
  # setting figure_size
  size = len(var_group)
  plt.figure(figsize = (7*size,5), dpi = 100)

  # for every variable
  for j,i in enumerate(var_group):
    norm_count = data[i].value_counts(normalize = True)
    n_uni = data[i].nunique()
  #Plotting the variable with every information
    plt.subplot(1,size,j+1)
    sns.barplot(norm_count, norm_count.index , order = norm_count.index)
    plt.xlabel('fraction/percent', fontsize = 20)
    plt.ylabel('{}'.format(i), fontsize = 20)
    plt.title('n_uniques = {} \n value counts \n {};'.format(n_uni,norm_count))
    

#%%

def BVA_categorical_plot(data, tar, cat):
  '''
  take data and two categorical variables,
  calculates the chi2 significance between the two variables 
  and prints the result with countplot & CrossTab
  '''
  #isolating the variables
  data = data[[cat,tar]][:]

  #forming a crosstab
  table = pd.crosstab(data[tar],data[cat],)
  f_obs = np.array([table.iloc[0][:].values,
                    table.iloc[1][:].values])

  #performing chi2 test
  from scipy.stats import chi2_contingency
  chi, p, dof, expected = chi2_contingency(f_obs)
  
  #checking whether results are significant
  if p<0.05:
    sig = True
  else:
    sig = False

  #plotting grouped plot
  sns.countplot(x=cat, hue=tar, data=data)
  plt.title("p-value = {}\n difference significant? = {}\n".format(round(p,8),sig))

  #plotting percent stacked bar plot
  #sns.catplot(ax, kind='stacked')
  ax1 = data.groupby(cat)[tar].value_counts(normalize=True).unstack()
  ax1.plot(kind='bar', stacked='True',title=str(ax1))
  int_level = data[cat].value_counts()
  
#%%

# Data Modeling I - Pipeline

def PreProcessing_Pipeline(X,y):
    '''
    Perform transformation of data in the following steps: 
        - FIll NAs (Simple Imputer)
        - One hot encoding to categorical variables
        - Scaling using MinMaxScaler for numerical variables.
                
    Parameters
    ----------
    df : dataframe

    Returns
    -------
    df_transformed : dataframe with the transformations after the preprocessing. 
    
    '''    
    cols_numerical = X.select_dtypes('float64').columns     
    cols_categorical = X.select_dtypes('category').columns   
    cols_numerical = cols_numerical.tolist()
    cols_categorical = cols_categorical.tolist(); #cols_categorical.remove('Loan_Status')
    
    # Define the pipeline steps (fill NAs, one hot, MinMaxScaler):
       
    numerical_preprocessor = make_pipeline(
        SimpleImputer(strategy="median"),
        MinMaxScaler()    
        )
    
    categorical_preprocessor = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(sparse=False, handle_unknown='ignore')
        )
       
    # define column transformer
    preprocessing = ColumnTransformer([
           ('numerical', numerical_preprocessor, cols_numerical),
           ('categorical', categorical_preprocessor, cols_categorical),
        ])
    
    # define pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessing)])
       
    # Fit and transform the data
    pipeline.fit(X,y)
    X_transformed = pipeline.named_steps['preprocessor'].transform(X)
    
    # Get the feature names after one-hot encoding
    feature_names = preprocessing.get_feature_names_out()     
      
    # Create a dataframe with the selected features
    X_transformed = pd.DataFrame(X_transformed, columns=feature_names)
        
    return preprocessing, X_transformed

#%%

# Data Modeling II - Machine Learning Models


def ML_Models(df):
    '''
    Function that trains different ML models doing different feature selection, hyperparameter tunning with a CV strategy.

    Parameters
    ----------
    df: dataframe with target variable (it must be df_transformed)

    Returns
    -------
    results : table with all results with
        - the ML model, 
        - feature selection method, 
        - number of features selected for feature selection (k), 
        - accuracy, f1 and confusion to evaluate result of the prediction of the ML model
        - best hyperparameters 
        
    '''
    #Define X, y
    X = df.drop(columns=['Loan_Status', 'Loan_ID'])
    y = df['Loan_Status'].replace({'Y': 1, 'N': 0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
    
    #Pre processing the data
    preprocessing, X_transformed = PreProcessing_Pipeline(X_train, y_train)
    
    # Define the models to evaluate
    models = []
    models.append(('SVC', SVC()))
    models.append(('LR', LogisticRegression()))
    models.append(('RF', RandomForestClassifier()))
        
    # Define the feature selection methods
    methods = [('KBest(chi2)', SelectKBest(chi2))]#, ('RFE', RFE(estimator=LogisticRegression()))] 
    
    # Define the number of features to consider
    k_list = [5,10]
    
    results = []
    
    # Evaluate models with 10-fold cross-validation using different feature selection methods and k values
    for name, model in models:
        print(f"Evaluating model {name}...")
        for method_name, method in methods:
            print(f"\tEvaluating feature selection method {method_name}...")
            for k in k_list:
                # Create the feature selection pipeline
                feature_selection = Pipeline([('pp', preprocessing), ('fs', method), ('k', SelectKBest(k=k))])                
                
                # Create the model pipeline
                model_pipeline = Pipeline([('fs', feature_selection), ('model', model)])
                
                # Define the cross-validation strategy
                kfold = KFold(n_splits=10, shuffle=True, random_state=17)
                
                # Fit the GridSearchCV object to the data
                if name == 'LR':
                    param_grid = {'model__C': [0.1, 1, 2, 3, 4, 5, 10, 20, 30, 50, 75, 90, 100], 'model__penalty': ['l1', 'l2']}
                    grid_search = GridSearchCV(model_pipeline, param_grid, cv=kfold)
                elif name == 'RF':
                    param_grid = {'model__n_estimators': [50, 100, 150],#, 200, 300, 400],
                                  'model__max_depth': [3, 5],#, 7, 9, 11],
                                  'model__min_samples_split': [2, 4],#, 6, 8],
                                  'model__min_samples_leaf': [1, 3],#, 5],
                                  'model__max_features': ['auto']}#], 'sqrt', 'log2']}
                    grid_search = GridSearchCV(model_pipeline, param_grid, cv=kfold)
                elif name == 'SVC':
                    param_grid = {'model__C': [0.1, 1, 10, 100, 1000], 
                                  'model__gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                                  'model__kernel': ['linear', 'rbf']}
                    grid_search = GridSearchCV(model_pipeline, param_grid, cv=kfold)
                                
                grid_search.fit(X_train, y_train)
                
                # Check if the best hyperparameters were found
                if hasattr(grid_search, 'best_estimator_'):
                    model_pipeline = grid_search.best_estimator_
                
                # Evaluate the model pipeline
                y_pred = cross_val_predict(model_pipeline, X_test, y_test, cv=kfold)
                
                # Calculate accuracy and F1 score
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                # Calculate confusion matrix
                conf_mat = confusion_matrix(y_test, y_pred)
                
                # Get the best hyperparameters for the model
                best_params = grid_search.best_params_
                
                # Store the results
                results.append({
                    'model': name,
                    'method': method_name,
                    'k': k,
                    'accuracy': accuracy,
                    #'selected_features': selected_features,
                    'f1': f1,
                    'confusion_matrix': conf_mat,
                    'best_params': best_params
                })
                print(f"Results for {name} model with {method_name} feature selection and k={k}:")
                print(results[-1]) # Print the last result added to the list
        
        print(f"{name} model finished.")                
    # Convert the results to a DataFrame
    results = pd.DataFrame(results)
    
    print("All models finished.")
    return results

#%%

# Extra - Create manual features

def create_manual_features (df):
    '''
    New variables are added to the dataframe df. 

    Parameters
    ----------
    df: dataframe (not transformed)

    Returns
    -------
    dff: dataframe with new features.
    
    '''
    dff = fill_NAs(df)
    encoder = LabelEncoder()
    #cols_categorical = dff.select_dtypes('category').columns
    #cols_categorical = cols_categorical.tolist(); cols_categorical.remove('Loan_Status')
    #for cat in cols_categorical:
    #    dff[cat] = encoder.fit_transform(dff[cat])
    
    dff['Loan_Amount_Term'] = dff['Loan_Amount_Term'].astype('int64')
    dff = dff[dff['Loan_Amount_Term'] > 0]
    dff['MF_Total_Income'] = dff['ApplicantIncome'] + dff['CoapplicantIncome']
    dff['MF_LoanAmount_log'] = np.log(dff['LoanAmount']); dff['MF_ApplicantIncome_log'] = np.log(dff['ApplicantIncome']);  dff['MF_Total_Income_log'] = np.log(dff['MF_Total_Income'])
    dff['MF_LoanAmount/AppIncome'] = dff['MF_LoanAmount_log'] / dff['MF_ApplicantIncome_log']
    dff['MF_LoanAmount/TotalInc'] = dff['MF_LoanAmount_log'] / dff['MF_Total_Income_log']
    dff['MF_LoanAmount_ratio'] = dff['LoanAmount']/dff['MF_Total_Income']
    dff['MF_Income_education'] = dff['MF_ApplicantIncome_log'] * encoder.fit_transform(dff['Education'])
    dff['MF_Income_employed'] = dff['MF_ApplicantIncome_log'] * encoder.fit_transform(dff['Self_Employed'])
    dff['MF_LoanAmount_monthly'] = dff['LoanAmount']/dff['Loan_Amount_Term']
       
    return dff

#%%

# Extra - Create auto features

def create_new_features(df, degree=2): #, n_bins=5
    '''
    Create a lot of new variables using polynomial, exponential, binning and sqrt transformation 
    
    Parameters
    ----------
    df : dataframe with or without transformation
    degree : degree of the polynomial transformation, by default 2

    Returns
    -------
    new_df : dataframe with new variables
        
    '''
    dff = fill_NAs(df)

    # Create polynomial features
    new_df_poly = pd.DataFrame()
    #new_df_sqrt = pd.DataFrame()
    #new_df_exp = pd.DataFrame()
    #new_df_bin = pd.DataFrame()
    
    #Encoding categorical variables
    encoder = LabelEncoder()
    cols_categorical = dff.select_dtypes('category').columns
    cols_categorical = cols_categorical.tolist(); cols_categorical.remove('Loan_Status')
    for cat in cols_categorical:
        dff[cat] = encoder.fit_transform(dff[cat])
    
    X = dff.drop(['Loan_Status', 'Loan_ID'], axis = 1)
    
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    new_df_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))
  
    #for col in X.select_dtypes(include = [np.float64]).columns:
        #new_df_sqrt['reciprocal_' + col] = 1/df[col]
        #new_df_exp['exp_' + col] = np.exp(df[col])
        #bins = np.linspace(df[col].min(), df[col].max(), n_bins + 1)
        #new_df_bin['bin_' + col] = pd.cut(df[col], bins)
        #new_df_bin['bin_' + col] = pd.get_dummies(new_df_bin['bin_' + col])    
    
    new_df = pd.concat([new_df_poly.reset_index(drop=True), dff], axis=1)
    #Remove duplicates
    new_df = new_df.loc[:,~new_df.T.duplicated(keep='first')]
    #Remove those variables with var=0
    # Get the variance of each column in the DataFrame
    variances = new_df.var()
    # Get the names of the columns with zero variance
    zero_variance_vars = variances[variances == 0].index.tolist()
    # Drop the columns with zero variance
    new_df.drop(columns=zero_variance_vars, inplace=True)
    #new_df.dropna(inplace=True)
    return new_df

#%%

def NN_Model (df):

    '''
    Train a simple neural network 
    
    Parameters
    ----------
    df: dataframe with target variable (it must be df_transformed)
    
    Returns
    -------
    The test accuracy, f1 score, and confusion matrix of the model
    
    '''
        
    #Define X, y
    X = df.drop(['Loan_Status'], axis = 1)
    y = df['Loan_Status'].replace({'Y': 1, 'N': 0})
    
    preprocessing, X_transformed = PreProcessing_Pipeline(X,y)
    
    # Train and test split
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=0)

    # Initialize the model
    model = Sequential()    
    # Add the first hidden layer with 128 neurons and a ReLU activation function
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    # Add the second hidden layer with 64 neurons and a ReLU activation function
    model.add(Dense(64, activation='relu'))
    # Add the output layer with 1 neuron and a sigmoid activation function for binary classification
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Fit the model to the training data
    model.fit(X_train, y_train, epochs=20, batch_size=32)
    # Predict the labels for the test data
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    # Calculate the test accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {test_accuracy}')
    # Calculate the F1 score
    f1 = f1_score(y_test, y_pred)
    print(f'F1 score: {f1}')
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f'Confusion matrix:\n{cm}')
    return test_accuracy, f1, cm    

#%%
# Functions not used
#Fill NA function

def fill_NAs (df):
    '''
    Replacing in categorical var. with most frequent values and in numerical with median. 
    '''
    
    cols_numerical = df.select_dtypes('float64').columns
    for col in cols_numerical:
        df[col].fillna((df[col].median()), inplace=True) 
    
    cols_categorical = df.select_dtypes('category').columns
    for col in cols_categorical:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df

#%%