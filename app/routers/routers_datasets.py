from fastapi import APIRouter, Query

from app.utils.utils_general import get_local_storage_path, load_data_from_csv
from app.utils.utils_hypothesis import DataframeImputation
import json
from starlette.responses import JSONResponse
import pandas as pd
from datetime import datetime

router = APIRouter()

@router.get("/Dataframe_preparation")
async def Dataframe_preparation(workflow_id: str,
                                step_id: str,
                                run_id: str,
                                file:str,
                                variables: list[str] | None = Query(default=None),
                                method: str | None = Query("mean",
                                                  regex="^(mean)$|^(median)$|^(most_frequent)$|^(constant)$|^(KNN)$|^(iterative)$")):

    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    # test_status = ''
    # dfv = pd.DataFrame()
    print(path_to_storage)
    try:
        test_status = 'Dataset is not defined'
        if file is None:
            test_status = 'Dataset is not defined'
            raise Exception
        test_status = 'Unable to retrieve datasets'
        # We expect only one here
        data = load_data_from_csv(path_to_storage + "/" + file)
        for variable in variables:
            if variable not in data.columns:
                raise Exception(str(variable) + '- The selected variable cannot be found in the dataset.')
        print(method)
        print(variables)

        x = DataframeImputation(data,variables,method)
        if type(x) == str:
            test_status= 'Failed to impute values'
            raise Exception (x)
        x.to_csv(path_to_storage + '/output/imputed_dataset.csv', index=False)

        df = pd.DataFrame(x.describe())
        df.insert(0, 'Index', df.index)
        df1 = pd.DataFrame()
        df1['Non Null Count'] = x.notna().sum()
        df1['Dtype'] = x.dtypes
        dfinfo = df1.T
        dfinfo['Index'] = dfinfo.index
        df = pd.concat([df, dfinfo], ignore_index=True)
        print(df)
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                    "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "step_id": step_id,
                    "test_name": method,
                    "test_params": variables
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = [{"file": 'expertsystem/workflow/' + workflow_id + '/' + run_id + '/' +
                                             step_id+'/analysis_output/' + 'imputed_dataset.svg'}]
            file_data["Saved_plots"]= []
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'newdataFrame': df.to_json(orient='records', default_handler=str)},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status + "\n" + e.__str__(),
                             'newdataFrame': '[]'},
                    status_code=200)
