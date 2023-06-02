import hydra
from omegaconf import DictConfig

import sklearn.preprocessing as pp2
from sklearn.feature_extraction import FeatureHasher

#from category_encoders import TargetEncoder
import numpy as np
from pycaret.internal.pipeline import Pipeline
from pycaret.internal.preprocess.preprocessor import PowerTransformer, StandardScaler, SimpleImputer, FixImbalancer, TransformerWrapper, TargetEncoder
from sklearn.linear_model._logistic import LogisticRegression
from imblearn.over_sampling import SMOTE

import pandas as pd
import preprocessors as pp

#Here you will put the selected pipeline from the notebook MLProfiling
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def datapipeline_new(cfg : DictConfig)-> Pipeline:
    datapipeline = Pipeline(
        steps = [
            (),
            (),
        ]
    )
    return datapipeline


#------------------------------------------
# --------- Below ... some examples -------
#
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def datapipeline4(cfg : DictConfig)-> Pipeline:
    #this pipeline is based mainly in Pycaret structure
    datapipeline= Pipeline(
        steps = [
                 ('keep_fields', 
                  pp.KeepNecessaryFeatures(variables_to_keep=cfg.data_pipeline.keep_features)),
                 
                 ('numerical_imputer',
                 TransformerWrapper(exclude=['match'],
                                    include=['dec_o', 'dec', 'like', 'attr',
                                             'prob', 'like_o', 'fun'],
                                    transformer=SimpleImputer(add_indicator=False,
                                                              copy=True,
                                                              fill_value=None,
                                                              keep_empty_features=False,
                                                              missing_values=np.nan,
                                                              strategy='mean',
                                                              verbose='deprecated'))),

                 ('categorical_imputer',
                 TransformerWrapper(exclude=['match'], include=['from', 'zipcode', 'undergra'],
                   transformer=SimpleImputer(add_indicator=False, copy=True,
                                             fill_value=None,
                                             keep_empty_features=False,
                                             missing_values=np.nan,
                                             strategy='most_frequent',
                                             verbose='deprecated'))),

                 ('rest_ecoding',
                 TransformerWrapper(exclude=['match'], include=['from', 'zipcode', 'undergra'],
                   transformer=TargetEncoder(cols=['from', 'zipcode',
                                                   'undergra'],
                                             drop_invariant=False,
                                             handle_missing='return_nan',
                                             handle_unknown='value',
                                             hierarchy=None,
                                             min_samples_leaf=20,
                                             return_df=True, smoothing=10,
                                             verbose=True))),

                 ('balance',
                 #TransformerWrapper(exclude=None,  include=None,
                 #  transformer=FixImbalancer(estimator=SMOTE(#k_neighbors=5,
                 #                                            #n_jobs=None,
                 #                                            #random_state=None,
                 #                                            sampling_strategy='auto'
                 #                                            )))
                    pp.Balancing_SMOTE_Encoding(label= cfg.data_fields.label)
                 ), 

                 ('debbuging', pp.Debbuging()
        
                 ),

                 ('transformation',
                 TransformerWrapper(exclude=['match'], include=None,
                   transformer=PowerTransformer(copy=False, method='yeo-johnson',
                                                standardize=False))),

                 ('normalize',
                 TransformerWrapper(exclude=['match'], include=None,
                   transformer=StandardScaler(copy=False, with_mean=True,
                                              with_std=True))),                                          
                  ],
    verbose=True)
    return datapipeline

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def datapipeline3(cfg : DictConfig)-> Pipeline:
    #this pipeline is based mainly in tailored preprocessors  transformers
    # all in preprocessors.py
    datapipeline = Pipeline(
        [
            ('keep_fields', 
             pp.KeepNecessaryFeatures(variables_to_keep=cfg.data_pipeline.keep_features)),
            ('fulfill_NAN', 
             pp.Fullfill_NAN()),
            ('standarize_text', 
             pp.Standarize_Text(variables_to_standarize_text=cfg.data_pipeline.standarize_text)),
            ('make_categorical', 
             pp.Make_Categorical(variables_to_categorize=cfg.data_pipeline.to_categorize)),
            ('convert_str_float', 
             pp.Convert_Str_to_Float(variables_to_convert=cfg.data_pipeline.str_to_float)),
            ('categorical_to_num', 
             pp.Categorical_to_Num(variables_to_convert=cfg.data_pipeline.category_to_num,
                                    file_dir = cfg.paths.interim_data)),
            ('onehot_encoding', 
             pp.OneHot_Encoding(variables_to_convert=cfg.data_pipeline.onehot_encoding)),
            
            ('drop_fields', pp.DropUnecessaryFeatures(variables_to_drop=cfg.data_pipeline.drop_features)),
          
            #('scaler', pp2.MinMaxScaler()),
        ]
    )
    return datapipeline


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def datapipeline2(cfg : DictConfig)-> Pipeline:
    #this pipeline is based mainly in skitlearn transformers
    dataPipeline = Pipeline(
       [
           ('scaler', pp2.StandardScaler(),)
           #('label_encoder', pp2.LabelEncoder()),
           #('one_hot_encoder', pp2.OneHotEncoder()),
           #('min_max_scaler', pp2.MinMaxScaler()),
           
       ] 
    )
    return dataPipeline

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def datapipeline(cfg : DictConfig)-> Pipeline:
    #this pipeline is based mainly in tailored preprocessors transformers as datapipeline3
    # all in preprocessors.py
    dataPipeline = Pipeline(

        [
            ('categorical_to_numerical',
                pp.CategoricalToNumerical(variables=cfg.data_pipeline.numerical_vars_from_numerical)),

            ('numerical_imputer',
                pp.NumericalImputer()),

            ('categorical_imputer',
               pp.CategoricalImputer(variables=cfg.data_pipeline.categorical_vars)),
             
            #('temporal_variable',
            #    pp.TemporalVariableEstimator(
            #       variables=cfg.TEMPORAL_VARS,
            #        reference_variable=cfg.DROP_FEATURES)),
            ('label extraction',
                pp.LabelExtraction(variables=cfg.data_pipeline.categorical_label_extraction)), 
            
            ('rare label encoder',
                pp.RareLabelCategoricalEncoder(
                    tol=0.01,
                    variables=cfg.data_pipeline.categorical_vars)),
             
            ('categorical_encoder',
               pp.CategoricalEncoder(variables=cfg.data_pipeline.categorical_vars)),
             
            #('feature hashing',
             #   FeatureHasher(n_features=10, input_type='string')),

            #('log_transformer',
            #    pp.LogTransformer()),
             
            ('scaler', pp2.MinMaxScaler()),

           # ('classifier', dummy())

            ('drop_fatures',
                pp.DropUnecessaryFeatures(variables_to_drop=cfg.data_pipeline.drop_features)),
            
            
           
        ]
    )

    return dataPipeline

