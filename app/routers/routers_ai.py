from fastapi import APIRouter, Request
from app.utils.mri_experiments import run_experiment
from app.utils.mri_grad_cam import visualize_grad_cam
from app.utils.mri_ig import visualize_ig
from app.utils.mri_deeplift import visualize_dl
from app.utils.mri_ggc import visualize_ggc

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
        slice: int
       ) -> dict:
    results = visualize_ig(model_path,
                           mri_path,
                           heatmap_path,
                           heatmap_name,
                           slice)
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
