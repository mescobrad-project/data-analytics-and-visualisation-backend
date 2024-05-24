import pandas as pd
from fastapi import APIRouter, Request, Query
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import shap

import pickle
import matplotlib.pyplot as plt
from starlette.responses import JSONResponse

from app.utils.mri_experiments import run_experiment
from app.utils.mri_ig import visualize_ig
from app.utils.mri_deeplift import visualize_dl
from app.utils.mri_ggc import visualize_ggc
from app.utils.utils_ai import train_linear_regression
from app.utils.utils_general import get_local_storage_path, load_data_from_csv

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
        test_status = ''
        data = load_data_from_csv(path_to_storage + "/" + file_name)
        X = data[independent_variables]
        y = data[dependent_variable]
        # Split dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
        linear_model = train_linear_regression(X_train, y_train)
        filename = model_name+'.sav'
        pickle.dump(linear_model, open(path_to_storage +'/'+ filename, 'wb'))

        # Make predictions
        y_pred = linear_model.predict(X_test)


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

        return JSONResponse(content={'status': 'Success', "mse": mse, "r2_score": r2_score_val, "Loss":loss, "mae":mae, "rmse":rmse,
                "coeff_determination":r_sq, 'intercept': linear_model.intercept_, 'slope': dfslope.transpose().to_json(orient='records')},
                                status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': '', "mse": '', "r2_score": '', "Loss":'', "mae":'', "rmse":'',
                "coeff_determination":'', 'intercept': '', 'slope': []},
                                status_code=200)
    # return {"mse": mse, "r2_score": r2_score_val, "Loss":loss, "coeff_determination":r_sq, 'intercept': linear_model.intercept_,'slope': linear_model.coef_}
