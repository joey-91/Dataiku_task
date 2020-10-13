import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

def convertAge(x):
    """
    creating age bandings, rather than integer values
    """
    if x < 20:
        return '<20'
    elif x < 25:
        return '20 - 25'
    elif x < 30:
        return '25 - 30'
    elif x < 35:
        return '30 - 35'
    elif x < 40:
        return '35 - 40'
    elif x < 45:
        return '40 - 45'
    elif x < 50:
        return '45 - 50'
    elif x < 55:
        return '50 - 55'
    elif x < 60:
        return '55 - 60'
    elif x < 70:
        return '60 - 70'
    elif x < 80:
        return '70 - 80'
    elif x < 120:
        return '80+'
    else:
        return np.nan

def childrenOrNotInUniverse(x):
    """
    To exclude children and datapoints with missing info from model. 
    With undesirable values set to NaN, I can easily drop them with pd.dropna().
    """
    try:
        if x.strip() in ['Not in universe', 'Not in universe or children', 'Children']:
            return np.nan 
        else:
            return x
    except:
        return x

def convertSalary(x):
    """
    Turning target variable into a binary event
    """
    if x == ' 50000+.':
        return 1
    else:
        return 0
    
def bar_charts(x, data, hue):
    """
    For Exploratory Analysis
    """
    plt.figure(figsize=(6,4))
    ax = sns.countplot(x=x, hue=hue, data=data)
    plt.xticks(rotation=90)
    plt.show()
    return ax

def convertChildren(x):
    """
    Grouping everyone with below 12th grade education level as children.
    This makes an assumption that everyone made it through school.
    """
    if x.strip() in ['10th grade', 'Children', 'Less than 1st grade', '7th and 8th grade',
                    '5th or 6th grade', '11th grade', '9th grade', '1st 2nd 3rd or 4th grade']:
        return 'Children'
    else:
        return x
    
def dataPrep(data, variables, upsample=True, drop_first=False, scale=True):
    """
    Function to prepare train & test datasets.
    Upsampling the data where salary is >$50,000 as there is a class imbalance.
    """
    data = data.copy()
    data = data[variables]
    
    if upsample:
        data_under50 = data[data.salary == 0]
        data_over50 = data[data.salary == 1]

        upsampled = resample(data_over50, 
                            replace=True,                # sample with replacement
                            n_samples=len(data_under50), # to match under 50 class
                            random_state=123) 

        data = pd.concat([data_under50, upsampled])         
    else:
        pass
    
    data = pd.get_dummies(data, drop_first=drop_first)
    
    # Creating Train and Test data
    train_x = data[data['source_train'] == 1].drop(['source_train', 'salary'], axis=1)
    test_x = data[data['source_train'] == 0].drop(['source_train', 'salary'], axis=1)
    train_y = data[data['source_train'] == 1]['salary']
    test_y = data[data['source_train'] == 0]['salary']
    
    #Scaling data
    if scale:
        scaler = StandardScaler()
        train_x = pd.DataFrame(scaler.fit_transform(train_x), 
                               columns=train_x.columns)
        test_x = pd.DataFrame(scaler.transform(test_x), 
                              columns=test_x.columns)      
    else:
        pass
    
    # Using this list of names for coefficient visualisation later
    column_names = list(data.columns)
    column_names.remove('source_train')
    column_names.remove('salary')
    
    return train_x, test_x, train_y, test_y, column_names