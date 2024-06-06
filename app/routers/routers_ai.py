from fastapi import APIRouter, Request
from app.utils.mri_experiments import run_experiment
from app.utils.mri_deeplift import visualize_dl
#from app.utils.mri_ggc import visualize_ggc
from app.utils.testing import mri_prediction, mris_batch_prediction

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
    results = run_experiment(data_path,
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
    results = visualize_dl(model_path,
                           mri_path,
                           heatmap_path)
    return {"results": results}

'''
@router.get("/ggc_explanation_experiment")
async def ggc_explanation_experiment(
        workflow_id: str,
        step_id: str,
        run_id: str,
        model_path: str,
        mri_path: str,
        heatmap_path: str,
        heatmap_name: str,
        axis: str,
        slice_idx: int
       ) -> dict:
    results = visualize_ggc(model_path,
                            mri_path,
                            heatmap_path,
                            heatmap_name,
                            axis,
                            slice_idx)
    return {"results": results}
'''

'''
@router.get("/mri_inference")
async def mri_inference(
        workflow_id: str,
        step_id: str,
        run_id: str,
        model_path: str,
        mri_path: str
       ) -> dict:
    results = mri_prediction(model_path,
                             mri_path)
    return {"results": results}
'''

@router.get("/mris_batch_inference")
async def mris_batch_inference(
        workflow_id: str,
        step_id: str,
        run_id: str,
        model_path: str,
        data_path: str,
        csv_path: str,
        output_path: str,
        batch_size: int
       ) -> dict:
    results = mris_batch_prediction(model_path,
                                    data_path,
                                    csv_path,
                                    output_path,
                                    batch_size)
    return {"results": results}
