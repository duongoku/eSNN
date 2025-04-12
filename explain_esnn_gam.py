import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import itertools
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from dataset.dataset import Dataset
from dataset.dataset_to_sklearn import SKLearnDataset, fromDataSetToSKLearn
from pygam import GAM, LogisticGAM, s
from utils.keras_utils import keras_sqrt_diff


def get_embeddings(model, X):
    return model.predict([X])


def fit_gam_model(X, y):
    n_features = X.shape[1]
    formula = s(0)
    for i in range(1, n_features):
        formula += s(i)

    gam = LogisticGAM(formula)  # formula = s(0) + ... + s(719)
    gam.gridsearch(X, y)

    return gam


def get_similarity_model(model):
    input1 = model.input[0]
    input2 = model.input[1]

    dist_output = model.get_layer("dist_output").output

    similarity_model = tf.keras.Model(inputs=[input1, input2], outputs=dist_output)

    return similarity_model


def plot_similarity(
    gam: GAM,
    output_dir: str,
    n_features: int,
    n_top_features: int,
    dataset: SKLearnDataset,
) -> pd.DataFrame:
    exit() #TODO: Remove
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature names
    feature_names = dataset.getDataFrame().columns.to_list()[:-2]
    
    importances = []
    
    for i, term in enumerate(gam.terms):
        if term.isintercept:
            continue
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
        
        # Plot partial dependence
        plt.figure()
        plt.plot(XX[:, term.feature], pdep)
        plt.fill_between(XX[:, term.feature], confi[:, 0], confi[:, 1], alpha=0.2)
        plt.title(f'Partial Dependence for {feature_names[term.feature]}')
        plt.xlabel(feature_names[term.feature])
        plt.ylabel('Partial Dependence')
        plt.savefig(os.path.join(output_dir, f'partial_dependence_feature_{term.feature}.png'))
        plt.close()
        
        # Compute importance as the range of partial dependence
        importance = np.max(pdep) - np.min(pdep)
        importances.append((feature_names[term.feature], importance))
    
    # Create DataFrame with feature importances
    importance_df = pd.DataFrame(importances, columns=['feature', 'importance'])
    importance_df = importance_df.sort_values(by='importance', ascending=False).head(n_top_features)
    
    return importance_df
    



def explain_similarity_kernel(
    dataset_name,
    full_model_path,
    output_dir,
    gam_object_file,
) -> pd.DataFrame | None:

    os.makedirs(output_dir, exist_ok=True)
    dataset = Dataset(dataset_name)
    dsl, _, _ = fromDataSetToSKLearn(dataset, True, n_splits=5)
    data = dsl.getFeatures()

    if os.path.exists(gam_object_file):
        with open(gam_object_file, "rb") as f:
            gam = pickle.load(f)
        gam.summary()
        importance_df = plot_similarity(
            gam=gam,
            output_dir=output_dir,
            n_features=data.shape[1],
            n_top_features=20,
            dataset=dsl,
        )
        return importance_df

    tf.keras.utils.get_custom_objects().update(
        {"Custom>keras_sqrt_diff": keras_sqrt_diff}
    )
    model = tf.keras.models.load_model(full_model_path)

    similarity_model = get_similarity_model(model)

    pairs = np.array(list(itertools.combinations(data.tolist(), 2)))
    
    similarities = similarity_model.predict([pairs[:, 0], pairs[:, 1]])

    feature_diffs = np.abs(pairs[:, 0] - pairs[:, 1])

    gam = fit_gam_model(feature_diffs, similarities)

    gam.summary()

    with open(gam_object_file, "wb") as f:
        pickle.dump(gam, f)
        print(f"Saved pyGAM model to {gam_object_file}")

    return explain_similarity_kernel(
        dataset_name,
        full_model_path,
        output_dir,
        gam_object_file,
    )



