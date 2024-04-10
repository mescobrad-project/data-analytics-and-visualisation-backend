from fastapi import APIRouter, Request
from app.utils.mri_experiments import run_experiment
from app.utils.mri_lrp_explanations import lrp_explanation

router = APIRouter()

@router.get("/ai_experiment")
async def ai_experiment(
        workflow_id: str,
        step_id: str,
        run_id: str,
        participants_path: str ,
        data_path: str,
        iterations : int = 5,
        model_type: str = "small",
        batch_size: int = 16,
        eval_size: int = 8,
        lr: float = 0.001,
        patience: int = 3,
       ) -> dict:

    results= run_experiment(iterations,
                   participants_path,
                   data_path,
                   model_type,
                   batch_size,
                   eval_size,
                   lr,
                   patience)
    return {"results": results}
    # files = get_files_for_slowwaves_spindle(workflow_id, run_id, step_id)
    # path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)

@router.get("/explanation_experiment")
async def explanation_experiment(
        workflow_id: str,
        step_id: str,
        run_id: str,
        model_path: str,
        mri_path: str,
        mri_slice: str,
        output_file_path: str,
        vmin: int = 90,
        vmax: float = 99.9,
        label: str | None = None,
       ) -> dict:
    results = lrp_explanation(
                        model_path,
                        mri_path,
                        mri_slice,
                        output_file_path,
                        label= label,
                        vmin=vmin, vmax=vmax)
    return {"results": results}
