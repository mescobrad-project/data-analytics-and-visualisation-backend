from fastapi import APIRouter, Request
from app.utils.mri_experiments import run_experiment
from app.utils.mri_grad_cam import visualize_grad_cam
from app.utils.mri_ig import visualize_ig

router = APIRouter()

@router.get("/ai_experiment")
async def ai_experiment(
        workflow_id: str,
        step_id: str,
        run_id: str,
        participants_path: str ,
        data_path: str,
        iterations : int = 5,
        batch_size: int = 16,
        eval_size: int = 8,
        lr: float = 0.001,
        patience: int = 3,
       ) -> dict:

    results= run_experiment(iterations,
                   participants_path,
                   data_path,
                   batch_size,
                   eval_size,
                   lr,
                   patience)
    return {"results": results}
    # files = get_files_for_slowwaves_spindle(workflow_id, run_id, step_id)
    # path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)

@router.get("/grad_cam_explanation_experiment")
async def grad_cam_explanation_experiment(
        workflow_id: str,
        step_id: str,
        run_id: str,
        model_path: str,
        mri_path: str,
        heatmap_path: str,
        heatmap_name: str
       ) -> dict:
    results = visualize_grad_cam(model_path,
                                 mri_path,
                                 heatmap_path,
                                 heatmap_name)
    return {"results": results}

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
