from fastapi import APIRouter, Request

# for tabular data - dense nn and ae models
from app.utils.tabular_nn_experiments import tabular_run_experiment
from app.utils.tabular_dnn_explanations import iXg_explanations

# for mris - conv3d model
from app.utils.mri_experiments import mri_run_experiment
from app.utils.mri_testing import mris_batch_prediction
from app.utils.mri_deeplift import visualize_dl

router = APIRouter()

@router.get("/ai_mri_training_experiment")
async def ai_mri_training_experiment(
        workflow_id: str,
        step_id: str,
        run_id: str,
        data_path: str,
        csv_path: str,
        iterations : int,
        batch_size: int,
        lr: float,
        early_stopping_patience: int
       ) -> dict:
    """ MRI Training Function To Be Implemented"""
    results = mri_run_experiment(data_path,
                                 csv_path,
                                 iterations,
                                 batch_size,
                                 lr,
                                 early_stopping_patience)
    return {"results": results}
    # files = get_files_for_slowwaves_spindle(workflow_id, run_id, step_id)
    # path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)


@router.get("/dl_explanation_experiment")
async def dl_explanation_experiment(
        workflow_id: str,
        step_id: str,
        run_id: str,
        model_path: str,
        mri_path: str,
        heatmap_path: str
       ) -> dict:
    """ MRI Explainability Function To Be Implemented"""

    results = visualize_dl(model_path,
                           mri_path,
                           heatmap_path)
    return {"results": results}


@router.get("/mris_batch_inference")
async def mris_batch_inference(
        workflow_id: str,
        step_id: str,
        run_id: str,
        model_path: str,
        data_path: str,
        csv_path: str,
        output_path: str
) -> dict:
    """ MRI Batch Inference Function To Be Implemented"""

    results = mris_batch_prediction(model_path,
                                    data_path,
                                    csv_path,
                                    output_path)
    return {"results": results}

@router.get("/ai_tabular_dnn_training_experiment")
async def ai_tabular_dnn_training_experiment(
        workflow_id: str,
        step_id: str,
        run_id: str,
        csv_path: str,
        no_of_features: int,
        test_size: float,
        iterations: int,
        lr: float,
        early_stopping_patience: int
       ) -> dict:

    model_type = 'dense_neural_network'
    results = tabular_run_experiment(csv_path,
                                     no_of_features,
                                     test_size,
                                     model_type,
                                     iterations,
                                     lr,
                                     early_stopping_patience)
    return {"results": results}

@router.get("/iXg_explanation_experiment")
async def iXg_explanations(
        workflow_id: str,
        step_id: str,
        run_id: str,
        model_path: str,
        csv_path: str,
       ) -> dict:

    results = iXg_explanations(model_path,
                               csv_path)
    return {"results": results}

@router.get("/ai_tabular_ae_training_experiment")
async def ai_tabular_ae_training_experiment(
        workflow_id: str,
        step_id: str,
        run_id: str,
        csv_path: str,
        no_of_features: int,
        test_size: float,
        iterations: int,
        lr: float,
        early_stopping_patience: int
       ) -> dict:

    model_type = 'autoencoder'
    results = tabular_run_experiment(csv_path,
                                     no_of_features,
                                     test_size,
                                     model_type,
                                     iterations,
                                     lr,
                                     early_stopping_patience)
    return {"results": results}
