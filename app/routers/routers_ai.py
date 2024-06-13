import pandas as pd
from fastapi import APIRouter, Request, Query
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, mean_absolute_error, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import shap
from sklearn.preprocessing import LabelEncoder

import pickle
import json
import matplotlib.pyplot as plt
from starlette.responses import JSONResponse

from app.utils.mri_experiments import run_experiment
from app.utils.mri_ig import visualize_ig
from app.utils.mri_deeplift import visualize_dl
from app.utils.mri_ggc import visualize_ggc
from app.utils.utils_ai import train_linear_regression, train_logistic_regression, train_SVC
from app.utils.utils_general import get_local_storage_path, load_data_from_csv
from datetime import datetime

router = APIRouter()

@router.get("/ai_mri_experiment")
async def ai_mri_experiment(
        workflow_id: str,
        step_id: str,
        run_id: str,
        participants_path: str,
        data_path: str,
        iterations : int = 5,
        batch_size: int = 4,
        eval_size: int = 30,
        lr: float = 0.001,
        es_patience: int = 3,
        scheduler_step_size: int = 3,
        scheduler_gamma: float = 0.75
       ) -> dict:

    results = run_experiment(iterations,
                   participants_path,
                   data_path,
                   batch_size,
                   eval_size,
                   lr,
                   es_patience,
                   scheduler_step_size,
                   scheduler_gamma
                   )
    return {"results": results}
    # files = get_files_for_slowwaves_spindle(workflow_id, run_id, step_id)
    # path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)

'''
@router.get("/grad_cam_explanation_experiment")
async def grad_cam_explanation_experiment(
        workflow_id: str,
        step_id: str,
        run_id: str,
        model_path: str,
        mri_path: str,
        heatmap_path: str,
        heatmap_name: str,
        slice: int,
        alpha: float
       ) -> dict:
    results = visualize_grad_cam(model_path,
                                 mri_path,
                                 heatmap_path,
                                 heatmap_name,
                                 slice,
                                 alpha)
    return {"results": results}
'''

@router.get("/ig_explanation_experiment")
async def ig_explanation_experiment(
        workflow_id: str,
        step_id: str,
        run_id: str,
        model_path: str,
        mri_path: str,
        heatmap_path: str,
        heatmap_name: str,
        slice: int,
        n_steps: int
       ) -> dict:
    results = visualize_ig(model_path,
                           mri_path,
                           heatmap_path,
                           heatmap_name,
                           slice,
                           n_steps)
    return {"results": results}


@router.get("/dl_explanation_experiment")
async def dl_explanation_experiment(
        workflow_id: str,
        step_id: str,
        run_id: str,
        model_path: str,
        mri_path: str,
        heatmap_path: str,
        heatmap_name: str,
        slice: int
       ) -> dict:
    results = visualize_dl(model_path,
                           mri_path,
                           heatmap_path,
                           heatmap_name,
                           slice)
    return {"results": results}

@router.get("/ggc_explanation_experiment")
async def ggc_explanation_experiment(
        workflow_id: str,
        step_id: str,
        run_id: str,
        model_path: str,
        mri_path: str,
        heatmap_path: str,
        heatmap_name: str,
        slice: int
       ) -> dict:
    results = visualize_ggc(model_path,
                            mri_path,
                            heatmap_path,
                            heatmap_name,
                            slice)
    return {"results": results}


@router.get("/linear_reg_create_model")
async def linear_reg_create_model(
        workflow_id: str,
        step_id: str,
        run_id: str,
        test_size: float,
        file_name:str,
        model_name:str,
        random_state:int,
        dependent_variable: str,
        shuffle:bool | None = Query(default=False),
        independent_variables: list[str] | None = Query(default=None)
) -> dict:

    try:
        df = pd.DataFrame()
        path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
        test_status = 'Unable to retrieve the dataset'
        data = load_data_from_csv(path_to_storage + "/" + file_name)
        X = data[independent_variables]
        y = data[dependent_variable]
        # Split dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
        test_status = 'Unable to execute regression'
        linear_model = train_linear_regression(X_train, y_train)

        test_status = 'Unable to save the model'
        filename = model_name+'.sav'
        pickle.dump(linear_model, open(path_to_storage +'/'+ filename, 'wb'))

        # TODO: Temporarily we save in both paths - to be available for Load model and also proceed to datalake
        pickle.dump(linear_model, open(path_to_storage +'/output/'+ filename, 'wb'))
        # Save the model parameters to a JSON file
        model_params = {
            'independent_variables': independent_variables,
            'dependent_variable': dependent_variable
        }
        with open(path_to_storage +'/'+ model_name+'.json', 'w') as f:
            json.dump(model_params, f)
        # Make predictions
        y_pred = linear_model.predict(X_test)

        test_status = 'Unable to print model stats'

        # The following section will get results by interpreting the created instance:
        # Obtain the coefficient of determination by calling the model with the score() function, then print the coefficient:
        r_sq = linear_model.score(X_train, y_train)
        # print('coefficient of determination:', r_sq)
        # loss = rmse(y_test, y_pred)
        loss = np.sqrt(np.mean(np.square(y_test - y_pred)))
        # print('Test Loss:', loss)
        # Calculate mean squared error
        mse = mean_squared_error(y_test, y_pred)
        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_test, y_pred)
        # Calculate R-squared (R²)
        r2_score_val = r2_score(y_test, y_pred)
        # print(r2_score_val)
        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)
        # plt.scatter(X_test, y_test, color="black")
        # plt.plot(X_test, y_pred, color="blue", linewidth=3)
        # plt.show()
        # Print the Intercept:
        # print('intercept:', linear_model.intercept_)
        # Print the Slope:
        dfslope= pd.DataFrame(linear_model.coef_.transpose(), index=independent_variables)
        # print('slope:', linear_model.coef_)
        # Predict a Response and print it:
        # y_pred = linear_model.predict(X_test)
        # print('Predicted response:', y_pred, sep='\n')
        test_status = 'Unable to present XAI plots'

        explainer = shap.LinearExplainer(linear_model, X_train, feature_names=independent_variables)
        shap_values = explainer(X_train)
        shap.summary_plot(shap_values, X_train, feature_names=independent_variables, show=False, max_display=20, plot_size=[8,5])
        plt.savefig(get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + "shap_summary_lr.svg", dpi=700)  # .png,.pdf will also support here
        # plt.show()
        plt.close()

        # shap.plots.waterfall(shap_values[sample_ind], max_display=14)
        shap.plots.waterfall(shap_values[1], max_display=20, show=False)
        plt.savefig(get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + "shap_waterfall_lr.svg",
                    dpi=700)  # .png,.pdf will also support here
        # plt.show()
        plt.close()
        shap.plots.heatmap(shap_values, show=False)
        plt.savefig(get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + "shap_heatmap_lr.svg",
                    dpi=700)  # .png,.pdf will also support here
        plt.close()
        # shap.plots.bar(shap_values, show=False, )
        # plt.savefig(get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + "shap_bar_lr.svg",
        #             dpi=700)  # .png,.pdf will also support here
        # # shap.plots.beeswarm(shap_values)
        # plt.close()
        shap.plots.violin(shap_values, show=False, plot_size=[8,5])
        plt.savefig(get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + "shap_violin_lr.svg",
                    dpi=700)  # .png,.pdf will also support here

        test_status = 'Error in creating info file.'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Linear Regression Create Model',
                "test_params": {
                    "Dependent variable": dependent_variable,
                    "Independent variables": independent_variables
                },
                "test_results": {"mse": mse, "r2_score": r2_score_val,
                                 "Loss": loss, "mae": mae, "rmse": rmse,
                                 'coeff_determination': r_sq,
                                 'intercept': linear_model.intercept_,
                                 'slope': dfslope.transpose().to_dict(),
                                 }}
            file_data['results'] = new_data
            file_data['Output_datasets'] = []
            file_data['Saved_plots'] = [{"file": 'expertsystem/workflow/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/analysis_output/shap_summary_lr.svg'},
                                        {"file": 'expertsystem/workflow/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/analysis_output/shap_waterfall_lr.svg'},
                                        {"file": 'expertsystem/workflow/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/analysis_output/shap_heatmap_lr.svg'},
                                         {"file": 'expertsystem/workflow/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/analysis_output/shap_violin_lr.svg'}
                                        ]
            file_data['Created_Model'] = [{"file": 'expertsystem/workflow/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/analysis_output/' + filename}]
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()


        return JSONResponse(content={'status': 'Success', "mse": mse, "r2_score": r2_score_val, "Loss":loss, "mae":mae, "rmse":rmse,
                    "coeff_determination":r_sq, 'intercept': linear_model.intercept_, 'slope': dfslope.transpose().to_json(orient='records')},
                                    status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, "mse": '', "r2_score": '', "Loss":'', "mae":'', "rmse":'',
                "coeff_determination":'', 'intercept': '', 'slope': []},
                                status_code=200)
    # return {"mse": mse, "r2_score": r2_score_val, "Loss":loss, "coeff_determination":r_sq, 'intercept': linear_model.intercept_,'slope': linear_model.coef_}


@router.get("/linear_reg_load_model")
async def linear_reg_load_model(
        workflow_id: str,
        step_id: str,
        run_id: str,
        file_name: str,
        model_name:str
) -> dict:

    try:
        path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
        test_status = 'Unable to retrieve the dataset'
        data = load_data_from_csv(path_to_storage + "/" + file_name)
        loaded_model = pickle.load(open(get_local_storage_path(workflow_id, run_id, step_id) +"/"+ model_name, 'rb'))
        with open(get_local_storage_path(workflow_id, run_id, step_id) +"/"+ model_name.replace('.sav','.json')) as f:
            test_params = json.load(f)
        X_test = data[test_params['independent_variables']]
        col_name = str(test_params['dependent_variable']) + '_predict'
        y_pred = pd.DataFrame(loaded_model.predict(X_test))
        print("Linear Predict")
        print(loaded_model.coef_)
        df = pd.DataFrame(loaded_model.coef_, index=test_params['independent_variables']).transpose()
        data.insert(loc=0, column=col_name, value=y_pred)
        result_dataset = data
        result_dataset.to_csv(path_to_storage + '/output/Dataset_predict.csv', index=False)

        test_status = 'Error in creating info file.'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Linear Regression Test Model',
                "test_params": {
                    "Dependent variable": test_params['dependent_variable'],
                    "Independent variables": test_params['independent_variables']
                },
                "test_results": {'coeff_determination': df.to_dict(),
                                 'intercept': loaded_model.intercept_,
                                 }}
            file_data['results'] = new_data
            file_data['Output_datasets'] = [{"file": 'expertsystem/workflow/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/analysis_output' + '/Dataset_predict.csv'}]
            file_data['Saved_plots'] = []
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()


        return JSONResponse(
            content={'status': 'Success', "coeff_determination": df.to_json(orient='records'),
                     'intercept': loaded_model.intercept_,
                     'dependent_param':test_params['dependent_variable'], 'independent_params':test_params['independent_variables'],
                     'result_dataset':result_dataset.to_json(orient='records')},
            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, "coeff_determination": '[]',
                     'intercept': '',
                     'dependent_param':'', 'independent_params':'','result_dataset':'[]'},
            status_code=200)


@router.get("/logistic_reg_create_model")
async def logistic_reg_create_model(
        workflow_id: str,
        step_id: str,
        run_id: str,
        test_size: float,
        file_name:str,
        model_name:str,
        random_state:int,
        dependent_variable: str,
        shuffle:bool | None = Query(default=False),
        independent_variables: list[str] | None = Query(default=None)
) -> dict:

    try:
        df = pd.DataFrame()
        path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
        test_status = 'Unable to retrieve the dataset'
        data = load_data_from_csv(path_to_storage + "/" + file_name)
        X = data[independent_variables]
        y = data[dependent_variable]

        print("here 1")
        # Split dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

        lab_enc = LabelEncoder()
        encoded = lab_enc.fit_transform(y_train)

        test_status = 'Unable to execute regression'
        logistic_model = train_logistic_regression(X_train, encoded)
        # logistic_model = train_logistic_regression(X_train, y_train)
        print("here 2")
        print(classification_report(y_test, logistic_model.predict(X_test)))

        test_status = 'Unable to save the model'
        filename = model_name+'.sav'
        pickle.dump(logistic_model, open(path_to_storage +'/'+ filename, 'wb'))

        # TODO: Temporarily we save in both paths - to be available for Load model and also proceed to datalake
        pickle.dump(logistic_model, open(path_to_storage +'/output/'+ filename, 'wb'))
        # Save the model parameters to a JSON file
        model_params = {
            'independent_variables': independent_variables,
            'dependent_variable': dependent_variable
        }
        with open(path_to_storage +'/'+ model_name+'.json', 'w') as f:
            json.dump(model_params, f)
        # Make predictions
        y_pred = logistic_model.predict(X_test)
        # encoded_pred = lab_enc.fit_transform(y_pred)
        test_status = 'Unable to print model stats'

        r_sq = logistic_model.score(X_train, encoded)
        # r_sq = logistic_model.score(X_train, y_train)
        print(r_sq)
        print(logistic_model.get_params())
        print(logistic_model.decision_function(X_train))
        print(logistic_model.coef_)
        mse = ""
        mae = ""
        r2_score_val = ""
        rmse = ""
        dfslope= pd.DataFrame(logistic_model.coef_.transpose(), index=independent_variables)
        test_status = 'Unable to present XAI plots'
        #
        explainer = shap.Explainer(logistic_model, X_train, feature_names=independent_variables)
        # shap_values = explainer(X_train)
        # shap.summary_plot(shap_values, X_train, feature_names=independent_variables, show=False, max_display=20, plot_size=[8,5])
        # plt.savefig(get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + "shap_summary_lr.svg", dpi=700)  # .png,.pdf will also support here
        # plt.close()
        # shap.plots.waterfall(shap_values[1], max_display=20, show=False)
        # plt.savefig(get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + "shap_waterfall_lr.svg",
        #             dpi=700)
        # plt.close()
        # shap.plots.heatmap(shap_values, show=False)
        # plt.savefig(get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + "shap_heatmap_lr.svg",
        #             dpi=700)
        # plt.close()
        #
        # shap.plots.violin(shap_values, show=False, plot_size=[8,5])
        # plt.savefig(get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + "shap_violin_lr.svg",
        #             dpi=700)

        test_status = 'Error in creating info file.'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Logistic Regression Create Model',
                "test_params": {
                    "Dependent variable": dependent_variable,
                    "Independent variables": independent_variables
                },
                "test_results": {"mse": mse, "r2_score": r2_score_val,
                                 "Loss": '', "mae": mae, "rmse": rmse,
                                 'coeff_determination': r_sq.to_dict(),
                                 'intercept': logistic_model.intercept_,
                                 'slope': dfslope.transpose().to_dict(),
                                 }}
            file_data['results'] = new_data
            file_data['Output_datasets'] = []
            file_data['Saved_plots'] = [{"file": 'expertsystem/workflow/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/analysis_output/shap_summary_lr.svg'},
                                        {"file": 'expertsystem/workflow/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/analysis_output/shap_waterfall_lr.svg'},
                                        {"file": 'expertsystem/workflow/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/analysis_output/shap_heatmap_lr.svg'},
                                         {"file": 'expertsystem/workflow/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/analysis_output/shap_violin_lr.svg'}
                                        ]
            file_data['Created_Model'] = [{"file": 'expertsystem/workflow/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/analysis_output/' + filename}]
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()


        return JSONResponse(content={'status': 'Success', "mse": mse, "r2_score": r2_score_val, "Loss":'', "mae":mae, "rmse":rmse,
                    "coeff_determination":r_sq, 'intercept': logistic_model.intercept_, 'slope': dfslope.transpose().to_json(orient='records')},
                                    status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, "mse": '', "r2_score": '', "Loss":'', "mae":'', "rmse":'',
                "coeff_determination":'', 'intercept': '', 'slope': []},
                                status_code=200)


@router.get("/logistic_reg_load_model")
async def logistic_reg_load_model(
        workflow_id: str,
        step_id: str,
        run_id: str,
        file_name: str,
        model_name:str
) -> dict:

    try:
        path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
        test_status = 'Unable to retrieve the dataset'
        data = load_data_from_csv(path_to_storage + "/" + file_name)
        print("Logistic LOAD")
        loaded_model = pickle.load(open(get_local_storage_path(workflow_id, run_id, step_id) +"/"+ model_name, 'rb'))
        print("Logistic LOADED")
        with open(get_local_storage_path(workflow_id, run_id, step_id) +"/"+ model_name.replace('.sav','.json')) as f:
            test_params = json.load(f)
        X_test = data[test_params['independent_variables']]
        col_name = str(test_params['dependent_variable']) + '_predict'
        y_pred = pd.DataFrame(loaded_model.predict(X_test))
        print("Logistic Predict")
        print(loaded_model.coef_)
        df = pd.DataFrame(loaded_model.coef_, index=test_params['independent_variables']).transpose()
        print("Logistic coefs_")
        data.insert(loc=0, column=col_name, value=y_pred)
        print("Logistic first column")
        result_dataset = data
        result_dataset.to_csv(path_to_storage + '/output/Dataset_predict.csv', index=False)

        test_status = 'Error in creating info file.'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Linear Regression Test Model',
                "test_params": {
                    "Dependent variable": test_params['dependent_variable'],
                    "Independent variables": test_params['independent_variables']
                },
                "test_results": {'coeff_determination': df.to_dict(),
                                 'intercept': loaded_model.intercept_,
                                 }}
            file_data['results'] = new_data
            file_data['Output_datasets'] = [{"file": 'expertsystem/workflow/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/analysis_output' + '/Dataset_predict.csv'}]
            file_data['Saved_plots'] = []
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()


        return JSONResponse(
            content={'status': 'Success', "coeff_determination": df.to_json(orient='records'),
                     'intercept': loaded_model.intercept_,
                     'dependent_param':test_params['dependent_variable'], 'independent_params':test_params['independent_variables'],
                     'result_dataset':result_dataset.to_json(orient='records')},
            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, "coeff_determination": '[]',
                     'intercept': '',
                     'dependent_param':'', 'independent_params':'','result_dataset':'[]'},
            status_code=200)


@router.get("/SVC_create_model")
async def SVC_create_model(
        workflow_id: str,
        step_id: str,
        run_id: str,
        test_size: float,
        file_name:str,
        model_name:str,
        random_state:int,
        dependent_variable: str,
        shuffle:bool | None = Query(default=False),
        independent_variables: list[str] | None = Query(default=None)
) -> dict:

    try:
        df = pd.DataFrame()
        path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
        test_status = 'Unable to retrieve the dataset'
        data = load_data_from_csv(path_to_storage + "/" + file_name)
        X = data[independent_variables]
        y = data[dependent_variable]
        # Split dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
        test_status = 'Unable to execute regression'
        SVC_model = train_SVC(X_train, y_train)

        test_status = 'Unable to save the model'
        filename = model_name+'.sav'
        pickle.dump(SVC_model, open(path_to_storage +'/'+ filename, 'wb'))

        # TODO: Temporarily we save in both paths - to be available for Load model and also proceed to datalake
        pickle.dump(SVC_model, open(path_to_storage +'/output/'+ filename, 'wb'))
        # Save the model parameters to a JSON file
        model_params = {
            'independent_variables': independent_variables,
            'dependent_variable': dependent_variable
        }
        with open(path_to_storage +'/'+ model_name+'.json', 'w') as f:
            json.dump(model_params, f)
        # Make predictions
        y_pred = SVC_model.predict(X_test)

        test_status = 'Unable to print model stats'

        r_sq = SVC_model.score(X_train, y_train)
        loss = np.sqrt(np.mean(np.square(y_test - y_pred)))
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2_score_val = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        dfslope= pd.DataFrame(SVC_model.coef_.transpose(), index=independent_variables)
        test_status = 'Unable to present XAI plots'

        explainer = shap.KernelExplainer(SVC_model.predict_proba, X_test, feature_names=independent_variables)
        shap_values = explainer.shap_values(X_test[0])
        shap.summary_plot(shap_values, X_train, feature_names=independent_variables, show=False, max_display=20, plot_size=[8,5])
        plt.savefig(get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + "shap_summary_lr.svg", dpi=700)  # .png,.pdf will also support here
        plt.close()
        shap.plots.waterfall(shap_values[1], max_display=20, show=False)
        plt.savefig(get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + "shap_waterfall_lr.svg",
                    dpi=700)
        plt.close()
        shap.plots.heatmap(shap_values, show=False)
        plt.savefig(get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + "shap_heatmap_lr.svg",
                    dpi=700)
        plt.close()

        shap.plots.violin(shap_values, show=False, plot_size=[8,5])
        plt.savefig(get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + "shap_violin_lr.svg",
                    dpi=700)

        test_status = 'Error in creating info file.'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Logistic Regression Create Model',
                "test_params": {
                    "Dependent variable": dependent_variable,
                    "Independent variables": independent_variables
                },
                "test_results": {"mse": mse, "r2_score": r2_score_val,
                                 "Loss": loss, "mae": mae, "rmse": rmse,
                                 'coeff_determination': r_sq.to_dict(),
                                 'intercept': SVC_model.intercept_,
                                 'slope': dfslope.transpose().to_dict(),
                                 }}
            file_data['results'] = new_data
            file_data['Output_datasets'] = []
            file_data['Saved_plots'] = [{"file": 'expertsystem/workflow/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/analysis_output/shap_summary_lr.svg'},
                                        {"file": 'expertsystem/workflow/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/analysis_output/shap_waterfall_lr.svg'},
                                        {"file": 'expertsystem/workflow/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/analysis_output/shap_heatmap_lr.svg'},
                                         {"file": 'expertsystem/workflow/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/analysis_output/shap_violin_lr.svg'}
                                        ]
            file_data['Created_Model'] = [{"file": 'expertsystem/workflow/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/analysis_output/' + filename}]
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()


        return JSONResponse(content={'status': 'Success', "mse": mse, "r2_score": r2_score_val, "Loss":loss, "mae":mae, "rmse":rmse,
                    "coeff_determination":r_sq, 'intercept': SVC_model.intercept_, 'slope': dfslope.transpose().to_json(orient='records')},
                                    status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, "mse": '', "r2_score": '', "Loss":'', "mae":'', "rmse":'',
                "coeff_determination":'', 'intercept': '', 'slope': []},
                                status_code=200)

@router.get("/SVC_load_model")
async def SVC_load_model(
        workflow_id: str,
        step_id: str,
        run_id: str,
        file_name: str,
        model_name:str
) -> dict:

    try:
        path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
        test_status = 'Unable to retrieve the dataset'
        data = load_data_from_csv(path_to_storage + "/" + file_name)
        loaded_model = pickle.load(open(get_local_storage_path(workflow_id, run_id, step_id) +"/"+ model_name, 'rb'))
        with open(get_local_storage_path(workflow_id, run_id, step_id) +"/"+ model_name.replace('.sav','.json')) as f:
            test_params = json.load(f)
        X_test = data[test_params['independent_variables']]
        col_name = str(test_params['dependent_variable']) + '_predict'
        y_pred = pd.DataFrame(loaded_model.predict(X_test))
        df = pd.DataFrame(loaded_model.coef_, index=test_params['independent_variables']).transpose()
        data.insert(loc=0, column=col_name, value=y_pred)
        result_dataset = data
        result_dataset.to_csv(path_to_storage + '/output/Dataset_predict.csv', index=False)

        test_status = 'Error in creating info file.'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'SVC Test Model',
                "test_params": {
                    "Dependent variable": test_params['dependent_variable'],
                    "Independent variables": test_params['independent_variables']
                },
                "test_results": {'coeff_determination': df.to_dict(),
                                 'intercept': loaded_model.intercept_,
                                 }}
            file_data['results'] = new_data
            file_data['Output_datasets'] = [{"file": 'expertsystem/workflow/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/analysis_output' + '/Dataset_predict.csv'}]
            file_data['Saved_plots'] = []
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()


        return JSONResponse(
            content={'status': 'Success', "coeff_determination": df.to_json(orient='records'),
                     'intercept': loaded_model.intercept_,
                     'dependent_param':test_params['dependent_variable'], 'independent_params':test_params['independent_variables'],
                     'result_dataset':result_dataset.to_json(orient='records')},
            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, "coeff_determination": '[]',
                     'intercept': '',
                     'dependent_param':'', 'independent_params':'','result_dataset':'[]'},
            status_code=200)
