from fastapi import APIRouter, Query

from app.utils.utils_general import get_local_storage_path, load_data_from_csv, get_all_files_from_local_temp_storage
from app.utils.utils_hypothesis import DataframeImputation
import json
from starlette.responses import JSONResponse
import pandas as pd
from datetime import datetime

router = APIRouter()

@router.get("/return_entire_dataset")
async def return_entire_dataset(workflow_id: str, step_id: str, run_id: str, file_name:str):
    try:
        path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
        name_of_files = get_all_files_from_local_temp_storage(workflow_id, run_id, step_id)
        if file_name in name_of_files:
            data = load_data_from_csv(path_to_storage + "/" + file_name)
        else:
            # print("Error : Failed to find the file")
            # return {'dataFrame': {}}
            raise Exception
        return JSONResponse(content={'status': 'Success', 'dataFrame': data.to_json(orient='records')},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': 'Error: '+ "\n" + e.__str__(),
                                     'dataFrame': '[]'},
                    status_code=200)

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
    # print(path_to_storage)
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
        x.to_csv(path_to_storage + '/output/imputed_'+file, index=False)
        # 12-06-2024
        # Changed in order to return the entire dataset
        # df = pd.DataFrame(x.describe())
        # df.insert(0, 'Index', df.index)
        # df1 = pd.DataFrame()
        # df1['Non Null Count'] = x.notna().sum()
        # df1['Dtype'] = x.dtypes
        # dfinfo = df1.T
        # dfinfo['Index'] = dfinfo.index
        # df = pd.concat([df, dfinfo], ignore_index=True)
        # print(df)
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
        return JSONResponse(content={'status': 'Success', 'newdataFrame': x.to_json(orient='records', default_handler=str)},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status + "\n" + e.__str__(),
                             'newdataFrame': '[]'},
                    status_code=200)


@router.get("/concat_csvs")
async def concat_csvs(workflow_id: str,
                                step_id: str,
                                run_id: str,
                                file1:str,
                                file2:str):

    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)

    try:
        print(file1)
        print(file2)
        test_status = 'Dataset is not defined'
        if (file1 or file2) is None:
            print('some null')
            test_status = 'Dataset is not defined'
            raise Exception
        test_status = 'Unable to retrieve datasets'
        # We expect only one here
        data1 = load_data_from_csv(path_to_storage + "/" + file1)
        data2 = load_data_from_csv(path_to_storage + "/" + file2)
        print(data2.columns)

        for column in data2.columns:
            newname=column+"_2"
            print(newname,"----", column)
            data2.rename(columns={str(column): str(newname)}, inplace=True)
            # column:column+'_2'
            # newcolumnsnames.append(column)
        # df.rename(columns={"A": "a", "B": "c"})
        print(data1.columns)
        print(data2.columns)
        df = pd.concat([data1, data2], axis=1)
        df.to_csv(path_to_storage + '/output/Datasets_concat.csv', index=False)
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                    "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "step_id": step_id,
                    "test_name": 'Concat Dataframes',
                    "test_params": [file1, file2]
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = [{"file": 'expertsystem/workflow/' + workflow_id + '/' + run_id + '/' +
                                             step_id+'/analysis_output/' + 'Datasets_concat.svg'}]
            file_data["Saved_plots"]= []
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'Datasets_concat': df.to_json(orient='records', default_handler=str)},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status + "\n" + e.__str__(),
                             'Datasets_concat': '[]'},
                    status_code=200)
