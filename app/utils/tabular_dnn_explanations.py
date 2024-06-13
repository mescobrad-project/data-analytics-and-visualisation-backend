import pandas as pd
import torch
import torch.nn as nn
from captum.attr import InputXGradient

from tabular_dnn import DenseNN

class ModelWrapper(nn.Module):

    def __init__(self, external_model):
        super(ModelWrapper, self).__init__()
        self.model = external_model

    def forward(self, x):
        _, logits = self.model(x)
        return logits


def iXg_explanations(model_path,
                     csv_path):

    '''

    Outputs
    - df: the dataframe of the original features
    - explanations_df: dataframe containing model prediction and iXg explanation of top predicted class

    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(model_path)
    wrapped_model = ModelWrapper(model)
    wrapped_model.to(device)
    wrapped_model.eval()  # Set model to evaluation mode

    df = pd.read_csv(csv_path)
    #df_cols = df.columns

    try:
        df = df.drop(['Unnamed: 0'], axis=1)
    except KeyError:
        pass

    # Move the column containing 'name' to the end if it exists
    name_col = None
    for col in df.columns:
        if 'label' in col:
            name_col = col
            break
    if name_col:
        cols = [col for col in df.columns if col != name_col]
        cols.append(name_col)
        df = df[cols]

    feature_names = [col for col in df.columns if col != 'label']
    explanations = []

    # Add prediction - probability - feature importance
    for index, row in df.iterrows():

        # Exclude the 'label' column to get the features
        features = row.drop('label').values
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device).requires_grad_()

        # Get model predictions
        logits = wrapped_model(x)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        top_prob, top_class = torch.max(probabilities, dim=1)

        # Get input gradient explanations
        ixg = InputXGradient(wrapped_model)
        attributions = ixg.attribute(x, target=top_class.item())

        # Convert attributions to a list
        attributions = attributions.squeeze().tolist()

        explanation_dict = {
            # 'index': index,
            'label': top_class.item(),
            'probability': f"{top_prob.item():.2f}"
        }

        # Add feature importances to the explanation_dict
        for feature_name, importance in zip(feature_names, attributions):
            explanation_dict[f'{feature_name} '] = f"{importance:.2f}"

        explanations.append(explanation_dict)

    df = df.round(2)

    for col in df.columns:
        if 'label' in col:
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].map(lambda x: f"{x:.2f}" if isinstance(x, (float, int)) else x)

    explanations_df = pd.DataFrame(explanations)
    explanations_df = explanations_df.round(2)

    def apply_styles(row):

        prediction_cols = ['label', 'probability']

        styles = []
        for col in row.index:
            if col in prediction_cols:
                styles.append('color: green')
            elif col in explanations_df.columns:
                styles.append('color: blue')
            else:
                styles.append('')
        return styles

    df = df.style.apply(apply_styles, axis=1)
    df = df.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
    df = df.set_properties(**{'text-align': 'center'})
    df = df.set_caption(
        "<div style='text-align: left; margin-bottom: 16px; font-size: 18px;'> Original features </div>")

    explanations_df = explanations_df.style.apply(apply_styles, axis=1)
    explanations_df = explanations_df.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
    explanations_df = explanations_df.set_properties(**{'text-align': 'center'})
    explanations_df = explanations_df.set_caption(
        "<div style='text-align: left; margin-bottom: 16px; font-size: 18px;'> Model prediction details (<span style='font-weight: bold; color: green;'>green</span>) and InputXGradient explanations (<span style='font-weight: bold; color: blue;'>blue</span>)</div>")

    return df, explanations_df
