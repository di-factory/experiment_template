import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
import os

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import random


def train(cfg : DictConfig)-> None:
    # Load the data
    train_features = pd.read_csv(os.path.join(cfg.paths.processed_data,
                                              cfg.file_names.train_features))
    train_labels = pd.read_csv(os.path.join(cfg.paths.processed_data, 
                                            cfg.file_names.train_labels))

    validation_features = pd.read_csv(os.path.join(cfg.paths.processed_data, 
                                                   cfg.file_names.validation_features))
    validation_labels = pd.read_csv(os.path.join(cfg.paths.processed_data, 
                                                 cfg.file_names.validation_labels))

    test_features = pd.read_csv(os.path.join(cfg.paths.processed_data, 
                                                   cfg.file_names.test_features))
    test_labels = pd.read_csv(os.path.join(cfg.paths.processed_data, 
                                                 cfg.file_names.test_labels))
    
    #test_labels.match = random.randint(0,1)

    mlflow.sklearn.autolog()

    for maxiter in [1000,1100]:
        with mlflow.start_run():
            # Create the model
            clf = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=maxiter,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=cfg.general_ml.seed, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)

            # Fit the model
            print('fitting the  model')
            print(train_features.shape, train_labels.shape)
            clf.fit(train_features, np.ravel(train_labels))

            # Make predictions

    
            print('making predictions')
            predictions = clf.predict(test_features)
            predict_proba = clf.predict_proba(test_features)
            signature = infer_signature(test_features, predictions)

            #for k, pred in enumerate(predictions):
            #print(f' {pred}, {predict_proba[k][0]:.6f},{predict_proba[k][1]:.6f} ')
            # Evaluate the model
            score= clf.score(test_features, test_labels)
            mlflow.log_metric("score", score)
            mlflow.sklearn.log_model(clf, 'MODEL',signature=signature)
            print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
        mlflow.end_run()

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig)-> None:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.tracking_experiment_name)
    train(cfg)
    
if __name__ == "__main__":
    main()





