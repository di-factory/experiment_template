import hydra
from omegaconf import DictConfig

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from imblearn.over_sampling import SMOTE
import os
import warnings

warnings.simplefilter('ignore', category=FutureWarning)

class Balancing_SMOTE_Encoding(BaseEstimator,TransformerMixin):
    """
     this preprocessor, corrects the imbalance classes, using SMOTE:
     
    """
    def __init__(self, label:DictConfig) -> None:
        self.sm = SMOTE(random_state=42)
        self.label = label
    
    def fit(self, X, y):
        self.sm.fit(X, X[self.label])
        return self

    def transform(self, X):
        print(f" .... INSIDE PIPELINE: Doing Balancing Encoding of {X.shape} rows ")
        X2, y2 = self.sm.fit_resample(X, X[self.label])
        print(f" .... DONE Balancing Encoding added rows upto: {X2.shape}")
        return X2, y2
    
class Debbuging(BaseEstimator,TransformerMixin):
    """
     this preprocessor is just for debbuging:
    """
    def __init__(self) -> None:
        print('debbuging init')
        pass

    def fit(self, X, y=None):
        print('debbuging fit:', X.shape, y.shape)
        return self

    def transform(self, X, y=None):
        print('debbuging transform:', X.shape, y.shape)
        return X, y
            


# ----------------------------
class OneHot_Encoding(BaseEstimator,TransformerMixin):
    """
     this preprocessor, Converts variables that are cathegorical to numerical and 
     saves a file with the conversion:
     
    """
    def __init__(self, variables_to_convert: DictConfig) -> None:
        self.variables_to_convert = variables_to_convert
        
    
    def fit(self, X, y=None):
        
        return self

    def transform(self, X):
        print(f" .... INSIDE PIPELINE: Doing Onehot Encoding of {self.variables_to_convert}: ")
        X = X.copy()
        for field in self.variables_to_convert:
            if X[field].dtype == 'category':
                enc = OneHotEncoder(sparse_output = False, 
                                    handle_unknown = "ignore", 
                                    max_categories = None)
                encoded = enc.fit_transform(pd.DataFrame(X[field]))
                # convert it to a dataframe
                
                encoded_df = pd.DataFrame(encoded, 
                                          columns=enc.get_feature_names_out())
               
                X = pd.concat([X,encoded_df], axis=1)
                
            else:
                print(f'{field } variable must be category. NOT TRASFORMED')
        return X
    

class Categorical_to_Num(BaseEstimator,TransformerMixin):
    """
     this preprocessor, Converts variables that are cathegorical to numerical and 
     saves a file with the conversion:
     
    """
    def __init__(self, variables_to_convert: DictConfig, file_dir: DictConfig) -> None:
        self.variables_to_convert = variables_to_convert
        self.file_dir = file_dir
    
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f" .... INSIDE PIPELINE: Doing Categorical to NUM Encoding of {self.variables_to_convert}: ")
        X = X.copy()
        for field in self.variables_to_convert:
            if X[field].dtype == 'category':
                le = LabelEncoder()
                X[f'{field}_encoded'] = le.fit_transform(X[field])
                pathfile = os.path.join(self.file_dir, 'cat2num.csv')
                pd.DataFrame(le.classes_, columns=[f'{field}_cat']).to_csv(pathfile, index=False)
            else:
                print(f'{field } variable must be category. NOT TRASFORMED')
        return X
    


class Convert_Str_to_Float(BaseEstimator,TransformerMixin):
    """
     this preprocessor, Converts any string to float:
     
    """
    def __init__(self, variables_to_convert: DictConfig) -> None:
        self.variables_to_convert = variables_to_convert
    
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f" .... INSIDE PIPELINE: Converting Strings to float of {self.variables_to_convert}: ")
        X = X.copy()
        for field in self.variables_to_convert:
            X[field]=X[field].str.replace(r',\d+|\d+,', '').astype('float64')
        return X
    

class Make_Categorical(BaseEstimator,TransformerMixin):
    """
     this preprocessor, Converts any kind of variable to categorical:
     
    """
    def __init__(self, variables_to_categorize: DictConfig) -> None:
        self.variables_to_categorize = variables_to_categorize
    
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f" .... INSIDE PIPELINE: Converting to categories: {self.variables_to_categorize}: ")
        X = X.copy()
        for field in self.variables_to_categorize:
            X[field] = X[field].astype('category')
        return X
    

class Fullfill_NAN(BaseEstimator, TransformerMixin):
    """
        this preprocessor, fullfills with np.NAN all the fields that are blank
    """

    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f" .... INSIDE PIPELINE: Fullfilling with NAN missing values in dataset ")
        X = X.copy()
        return X.fillna(np.NAN)
    
class Standarize_Text(BaseEstimator,TransformerMixin):
    """
     this preprocessor, standarize the text in fields passed:
     all in downcase(), special chars outside [a..z, 0..9], converts to '_'
    """
    def __init__(self, variables_to_standarize_text: DictConfig) -> None:
        self.variables_to_standarize_text = variables_to_standarize_text
    
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f" .... INSIDE PIPELINE: Standarizing to lower and _ of: {self.variables_to_standarize_text}: ")
        X = X.copy()
        for field in self.variables_to_standarize_text:
            X[field] = X[field].str.lower()
            X[field] = X[field].str.replace('[^a-z0-9]', '_')
        return X
    

class KeepNecessaryFeatures(BaseEstimator, TransformerMixin):
    """
        this preprocessor, eliminates all but the selected columns that come
        in variables_to_drop set
    """

    def __init__(self, variables_to_keep:DictConfig) -> None:
        self.variables_to_keep = variables_to_keep

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f" .... INSIDE PIPELINE: Dropping no needed vars, keeping only: {self.variables_to_keep}")
        X = X.copy()
        columns_to_drop = X.drop(self.variables_to_keep, axis = 1)
        return X.drop(columns_to_drop, axis=1).fillna(np.NAN)

class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):
    """
        this preprocessor, eliminates the selected columns that come
        in variables_to_drop set
    """

    def __init__(self, variables_to_drop: DictConfig=None) -> None:
        self.variables_to_drop = variables_to_drop

    def fit(self, X, y=None):
        
        return self


    def transform(self, X):
        print(f" .... INSIDE PIPELINE: Dropping this vars after transformed: {self.variables_to_drop}: ")
        X = X.copy()
        return X.drop(self.variables_to_drop, axis=1).fillna(np.NAN)


#__________________________
# categorical missing value imputer
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        print("init with Categorical Imputer")
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')

        return X


# Numerical missing value imputer
class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):        
        self.variables = [var for var in X.columns if X[var].dtype != 'O']
        # persist mode in a dictionary
        self.imputer_dict_ = {}
        
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mode()[0]
        return self

    def transform(self, X):
        print("init with Numerical Imputer")
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X


# Temporal variable calculator
'''
class TemporalVariableEstimator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None, reference_variable=None):
        
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.reference_variables = reference_variable

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[self.reference_variables] - X[feature]

        return X
'''

# transform categorical to numerical
class CategoricalToNumerical(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        def object_to_num(x):

            if x is not np.nan:
                x = ''.join(x.split(','))

            return x
        
        print("init with Categorical To Numerical")
        for feature in self.variables:
            X[feature] = X[feature].apply(object_to_num)
            X[feature] = X[feature].astype(np.float64)
        return X


# frequent label categorical encoder
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tol=0.05, variables=None):
        self.tol = tol        
        self.variables = variables

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts() / np.float64(len(X)))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self, X):
        print("init with Rare Label Categorical")
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(self.encoder_dict_[
                    feature]), X[feature], 'Rare')

        return X


# string to numbers categorical encoder
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None, target=None):
        self.variables = variables

    def fit(self, X, y=None):         
        self.enc = OneHotEncoder(handle_unknown='ignore')
        # persist transforming dictionary
        
        if len(self.variables) == 1:
            self.enc.fit(np.array(X[self.variables[0]]).reshape(-1,1))
        else:
            self.enc.fit(np.array(X[self.variables]))

        return self

    def transform(self, X):
        print("init with Strings to Number Categorical")
        # encode labels
        X = X.copy()
        
        if len(self.variables) == 1:
            X_enc = self.enc.transform(np.array(X[self.variables[0]]).reshape(-1,1)).toarray()


        else:
            X_enc = self.enc.transform(np.array(X[self.variables])).toarray()


        X_enc = pd.DataFrame(X_enc, columns = [self.variables[j] + '_' + str(i) for j in range(len(self.variables)) for i in range(len(self.enc.categories_[j]))])
        
        X = pd.concat([X, X_enc], axis=1).drop(self.variables, axis=1)
            
        return X


# logarithm transformer
class LogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
       self.variables = variables

    def fit(self, X, y=None):
        # to accomodate the pipeline
        self.variables = [var for var in X.columns if X[var].dtype != 'O']
        return self

    def transform(self, X):
        print("init with Log Transformer")
        X = X.copy()

        for feature in self.variables:
            X[feature] = np.log(X[feature])

        return X




class LabelExtraction(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("init with Label Extraction")
        X = X.copy()
        for feature in self.variables:
            for i in range(X.shape[0]):
                try:
                    X[feature][i] = X[feature][i].split(',')[0]
                except:
                    pass

        return X







