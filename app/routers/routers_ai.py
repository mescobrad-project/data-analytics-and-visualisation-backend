from fastapi import APIRouter, Request
from app.utils.mri_experiments import run_experiment
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
    return true
    # files = get_files_for_slowwaves_spindle(workflow_id, run_id, step_id)
    # path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
