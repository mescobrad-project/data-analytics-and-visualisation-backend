import csv
import json

import numpy as np
from fastapi import Query, APIRouter
import pyActigraphy
import os
import pandas as pd
from pyActigraphy.analysis import Cosinor
import plotly.graph_objects as go
from lmfit import fit_report
from datetime import datetime


import pandas
import plotly.graph_objs as go

# import plotly.graph_objs as go
from starlette.responses import JSONResponse

from app.utils.utils_general import get_local_storage_path

router = APIRouter()
@router.get("/return_actigraphy_data", tags=["actigraphy_data"])
async def return_actigraphy_data():
    with open('example_data/actigraphy_relevant_dataset.csv', newline="") as csvfile:
        if not os.path.isfile('example_data/actigraphy_relevant_dataset.csv'):
            return []
        reader = csv.reader(csvfile, delimiter=',')
        results_array = []
        i = 0
        for row in reader:
            i += 1
            temp_to_append = {
                "id": i,
                "line": row[0],
                "date": row[1],
                "time": row[2],
                "activity": row[3],
                "marker": row[4],
                "whitelight": row[5],
                "sleep_wake": row[6],
                "interval_status": row[7]
            }
            results_array.append(temp_to_append)
            # print(results_array)
        return results_array

    return 1

@router.get("/return_actigraphy_general_data", tags=["actigraphy_general_data"])
async def return_actigraphy_general_data():
    with open('example_data/actigraphy_general_relevant_dataset.csv', newline="") as csvfile:
        if not os.path.isfile('example_data/actigraphy_relevant_dataset.csv'):
            return []
        reader = csv.reader(csvfile, delimiter=',')
        results_array = []
        i = 0
        for row in reader:
            i += 1
            temp_to_append = {
                "id": i,
                "interval_type": row[0],
                "interval": row[1],
                "date_start": row[2],
                "time_start": row[3],
                "date_stop": row[4],
                "time_stop": row[5],
                "duration": row[6],
                "invalid_sw": row[7],
                "efficiency": row[8],
                "wake_time": row[9],
                "sleep_time": row[10],
                "sleep": row[11],
                "exposure_white": row[12],
                "average_white": row[13],
                "max_white": row[14],
                "talt_white": row[15],
                "invalid_white": row[16]
            }
            results_array.append(temp_to_append)
            print(results_array)
        return results_array

    return 1


@router.get("/return_cole_kripke", tags=["actigraphy_analysis_assessment_algorithm"])
async def return_cole_kripke():
     raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv', start_time='2022-07-18 12:00:00',
         period='1 day',
         language='ENG_UK'
     )
     layout = go.Layout(title="Rest/Activity detection", xaxis=dict(title="Date time"),
                        yaxis=dict(title="Counts/period"), showlegend=False)
     CK = raw.CK()
     layout.update(yaxis2=dict(title='Classification', overlaying='y', side='right'), showlegend=True);
     output = go.Figure(data=[
         go.Scatter(x=raw.data.index.astype(str), y=raw.data, name='Data'),
         go.Scatter(x=CK.index.astype(str), y=CK, yaxis='y2', name='CK')
     ], layout=layout)
     return output.show()
    # path = os.path.join(os.path.dirname(pyActigraphy.__file__), 'C:\\Users\\George Ladikos\\')
    # datetime_list = ['2022-07-18 12:00:00', '2022-07-19 12:00:00', '2022-07-20 12:00:00', '2022-07-21 12:00:00',
    #                  '2022-07-22 12:00:00', '2022-07-23 12:00:00', '2022-07-24 12:00:00', '2022-07-25 12:00:00']
    #
    # day_count = 1
    # for i in datetime_list:
    #     raw_data = pyActigraphy.io.read_raw_rpx(
    #         path + '0345-024_18_07_2022_13_00_00_New_Analysis_Ophir.csv', start_time=i, period='1 day',
    #         language='ENG_UK'
    #     )
    #     layout = go.Layout(title="Rest/Activity detection Day " + str(day_count), xaxis=dict(title="Date time"),
    #                        yaxis=dict(title="Counts/period"), showlegend=False)
    #     CK = raw_data.CK()
    #     layout.update(yaxis2=dict(title='Classification', overlaying='y', side='right'), showlegend=True);
    #     chart = go.Figure(data=[
    #         go.Scatter(x=raw_data.data.index.astype(str), y=raw_data.data, name='Data'),
    #         go.Scatter(x=CK.index.astype(str), y=CK, yaxis='y2', name='CK')
    #     ], layout=layout)
    #     chart.show()
    #     day_count = day_count + 1

@router.get("/return_sadeh_scripp", tags=["actigraphy_analysis_assessment_algorithm"])
async def return_sadeh_scripp():
     raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv', start_time='2022-07-18 12:00:00',
         period='1 day',
         language='ENG_UK'
     )
     layout = go.Layout(title="Rest/Activity detection", xaxis=dict(title="Date time"),
                        yaxis=dict(title="Counts/period"), showlegend=False)
     sadeh = raw.Sadeh()
     scripps = raw.Scripps()
     layout.update(yaxis2=dict(title='Classification', overlaying='y', side='right'), showlegend=True);
     output = go.Figure(data=[
                go.Scatter(x=raw.data.index.astype(str),y=raw.data, name='Data'),
                go.Scatter(x=sadeh.index.astype(str),y=sadeh, yaxis='y2', name='Sadeh'),
                go.Scatter(x=scripps.index.astype(str),y=scripps, yaxis='y2', name='Scripps')
            ], layout=layout)
     return output.show()

@router.get("/return_oakley", tags=["actigraphy_analysis_assessment_algorithm"])
async def return_oakley():
     raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv', start_time='2022-07-18 12:00:00',
         period='1 day',
         language='ENG_UK'
     )
     layout = go.Layout(title="Rest/Activity detection", xaxis=dict(title="Date time"),
                        yaxis=dict(title="Counts/period"), showlegend=False)
     oakley = raw.Oakley(threshold=40)
     oakley_auto = raw.Oakley(threshold='automatic')
     # sadeh = raw.Sadeh()
     # scripps = raw.Scripps()
     layout.update(yaxis2=dict(title='Classification', overlaying='y', side='right'), showlegend=True);
     output = go.Figure(data=[
                go.Scatter(x=raw.data.index.astype(str),y=raw.data, name='Data'),
                go.Scatter(x=oakley.index.astype(str),y=oakley, yaxis='y2', name='Oakley (thr: medium)'),
                go.Scatter(x=oakley_auto.index.astype(str),y=oakley_auto, yaxis='y2', name='Oakley (thr: automatic)')
            ], layout=layout)
     return output.show()

@router.get("/return_crespo", tags=["actigraphy_analysis_assessment_algorithm"])
async def return_crespo():
     raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv', start_time='2022-07-18 12:00:00',
         period='1 day',
         language='ENG_UK'
     )
     layout = go.Layout(title="Rest/Activity detection", xaxis=dict(title="Date time"),
                        yaxis=dict(title="Counts/period"), showlegend=False)
     crespo = raw.Crespo()
     crespo_6h = raw.Crespo(alpha='6h')
     crespo_zeta = raw.Crespo(estimate_zeta=True)
     # sadeh = raw.Sadeh()
     # scripps = raw.Scripps()
     layout.update(yaxis2=dict(title='Classification', overlaying='y', side='right'), showlegend=True);
     output = go.Figure(data=[
                go.Scatter(x=raw.data.index.astype(str),y=raw.data, name='Data'),
                go.Scatter(x=crespo.index.astype(str),y=crespo, yaxis='y2', name='Crespo'),
                go.Scatter(x=crespo_6h.index.astype(str),y=crespo_6h, yaxis='y2', name='Crespo (6h)'),
                go.Scatter(x=crespo_zeta.index.astype(str),y=crespo_zeta, yaxis='y2', name='Crespo (Automatic)')
            ], layout=layout)
     return output.show()

@router.get("/return_roenneberg", tags=["actigraphy_analysis_assessment_algorithm"])
async def return_roenneberg():
     raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv', start_time='2022-07-18 12:00:00',
         period='1 day',
         language='ENG_UK'
     )
     layout = go.Layout(title="Rest/Activity detection", xaxis=dict(title="Date time"),
                        yaxis=dict(title="Counts/period"), showlegend=False)
     roenneberg = raw.Roenneberg()
     roenneberg_thr = raw.Roenneberg(threshold=0.25, min_seed_period='15min')
     layout.update(yaxis2=dict(title='Classification', overlaying='y', side='right'), showlegend=True);
     output = go.Figure(data=[
                go.Scatter(x=raw.data.index.astype(str),y=raw.data, name='Data'),
                go.Scatter(x=roenneberg.index.astype(str),y=roenneberg, yaxis='y2', name='Roenneberg'),
                go.Scatter(x=roenneberg_thr.index.astype(str),y=roenneberg_thr, yaxis='y2', name='Roenneberg (Thr:0.25)')
            ], layout=layout)
     return output.show()

@router.get("/return_weekly_activity", tags=["actigraphy_analysis"])
async def return_weekly_activity():
    datetime_list = [
                     '2022-07-18 12:00:00', '2022-07-19 12:00:00', '2022-07-20 12:00:00', '2022-07-21 12:00:00',
                     '2022-07-22 12:00:00', '2022-07-23 12:00:00', '2022-07-24 12:00:00', '2022-07-25 12:00:00'
                    ]
    # raw = pyActigraphy.io.read_raw_rpx(
    #                                     'example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv',
    #                                     start_time='2022-07-18 12:00:00',
    #                                     period='1 day',
    #                                     language='ENG_UK'
    #                                   )
    day_count = 1
    for i in datetime_list:
        raw = pyActigraphy.io.read_raw_rpx(
            'example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv',
            start_time=i,
            period='1 day',
            language='ENG_UK'
        )
        layout = go.Layout(
            title="Actigraphy data weekly activity day " + str(day_count),
            xaxis=dict(title="Date time"),
            yaxis=dict(title="Counts/period"),
            showlegend=False
        )
        output = go.Figure(data=[go.Scatter(x=raw.data.index.astype(str), y=raw.data)], layout=layout)
        output.show()
        day_count = day_count + 1
    # raw.name
    # raw.start_time
    # raw.duration()
    # raw.uuid
    # raw.frequency

@router.get("/rest_to_activity_probability", tags=["actigraphy_analysis"])
async def rest_to_activity_probability():
    raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv', period='7days')

    # create objects for layout and traces
    layout = go.Layout(title="", xaxis=dict(title=""), showlegend=False)
    pRA, pRA_weights = raw.pRA(0, start='00:00:00', period='8H')
    layout.update(title="Rest->Activity transition probability", xaxis=dict(title="Time [min]"), showlegend=False);
    output = go.Figure(data=go.Scatter(x=pRA.index, y=pRA, name='', mode='markers'), layout=layout)
    return output.show()

@router.get("/sleep_diary", tags=["actigraphy_analysis"])
async def sleep_diary():
    raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv')
    return raw.start_time, raw.duration()
@router.get("/return_average_daily_activity", tags=["actigraphy_analysis"])
async def return_average_daily_activity():
    raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv')
    # raw.name
    # raw.start_time
    # raw.duration()
    # raw.uuid
    # raw.frequency
    layout = go.Layout(
        title="Actigraphy data weekly activity",
        xaxis=dict(title="Date time"),
        yaxis=dict(title="Counts/period"),
        showlegend=False
    )
    layout.update(title="Daily activity profile", xaxis=dict(title="Date time"), showlegend=False);
    daily_profile = raw.average_daily_activity(freq='15min', cyclic=False, binarize=False)
    output = go.Figure(data=[
                                go.Scatter(x=daily_profile.index.astype(str), y=daily_profile)
                            ], layout=layout
                      )
    return output.show()

# @router.get("/return_diary", tags=["actigraphy_analysis"])
# async def return_diary():
#     fpath = os.path.join(os.path.dirname(pyActigraphy.__file__), 'C:\\Users\\George Ladikos\\')
#     raw = pyActigraphy.io.read_raw_rpx(
#         fpath + '0345-024_18_07_2022_13_00_00_New_Analysis_Ophir.csv', start_time='2022-07-18 12:00:00', period='1 day',
#         language='ENG_UK'
#     )
#     layout = go.Layout(title="Rest/Activity detection", xaxis=dict(title="Date time"),
#                        yaxis=dict(title="Counts/period"), showlegend=False)
#     CK = raw.CK()
#     layout.update(yaxis2=dict(title='Classification', overlaying='y', side='right'), showlegend=True);
#     output = go.Figure(data=[
#         go.Scatter(x=raw.data.index.astype(str), y=raw.data, name='Data'),
#         go.Scatter(x=CK.index.astype(str), y=CK, yaxis='y2', name='CK')
#     ], layout=layout)
#     return output.show()

# async def return_diary():
#     xx = 0
#     try:
#         # raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/raw_sample.csv', 'FR', True, None, None, float, float, ';')
#         raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv')
#         print(raw.start_time)
#         print(raw.duration())
#         xx = raw.IS()
#         # raw = pyActigraphy.io.read_raw_rpx(fpath + 'raw_sample.csv')
#     except:
#         print("An exception occurred")
#
#     # raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/Combined Export File.csv', 'ENG_US')
#
#     # raw.start_time
#     # raw.duration()
#     # sleep_diary = raw.read_sleep_diary('example_data/actigraph/Neurophy_Actigraph.csv')
#     # raw.sleep_diary.name
#     # raw.sleep_diary.diary
#     # raw.sleep_diary.summary()
#     print(xx)
#     return 1

def return_rawObject():
    xx=0
    try:
        # fpath = f"C:/Users/gdoukas/PycharmProjects/data-analytics-and-visualisation-backend/example_data/actigraph/example_data/actigraph/Neurophy_Actigraph.csv"
        raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/test_sample.csv','FR')
        print(type(raw))
        # raw = pyActigraphy.io.read_raw_rpx(fpath, 'FR')
        xx = raw.IS()
    except:
        print("An exception occurred")

    # raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/Combined Export File.csv', 'ENG_US')

    # raw.start_time
    # raw.duration()
    # sleep_diary = raw.read_sleep_diary('example_data/actigraph/Neurophy_Actigraph.csv')
    # raw.sleep_diary.name
    # raw.sleep_diary.diary
    # raw.sleep_diary.summary()
    return xx


ww = return_rawObject()
print(ww)


@router.get("/actigraphy_metrics")
async def actigraphymetrics(workflow_id: str,
                            step_id: str,
                            run_id: str,
                            number_of_offsets: int,
                            number_of_periods: int,
                            file: str,
                            binarize: bool | None = Query(default=True),
                            period_offset: str | None = Query(default="Day",
                                                              regex="^(Day)$|^(Week)$|^(Month)$"),
                            threshold: int | None = Query(default=4),
                            # metric: str | None = Query(default='IS', enum=['IS','ISm','IV','IVm', 'L5','M10','RA','L5p','ADAT','ADATp','M10p','RAp','kRA','kAR']),
                            freq_offset: str | None = Query(default="Hour",
                                                              regex="^(Hour)$|^(Minute)$|^(Second)$")):
    test_status = ''
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    try:
        test_status = 'Unable to retrieve actigraphy file.'
        # raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv')
        raw = pyActigraphy.io.read_raw_rpx(path_to_storage + "/" + file)
        test_status = 'Unable to compute metrics.'

        if freq_offset == 'Hour':
            freq = f"{number_of_offsets}{pd.offsets.Hour._prefix}"
        elif freq_offset == 'Minute':
            freq = f"{number_of_offsets}{pd.offsets.Minute._prefix}"
        else:
            freq = f"{number_of_offsets}{pd.offsets.Second._prefix}"

        if period_offset == 'Day':
            period = f"{number_of_periods}{pd.offsets.Day._prefix}"
        elif period_offset == 'Week':
            period = f"{number_of_periods}{pd.offsets.Week._prefix}"
        else:
            period = f"{number_of_periods}{pd.offsets.MonthBegin._prefix}"

        print("Name"+ raw.name)
        print("Start_time"+ (raw.start_time).__str__())
        print("Duration"+ (raw.duration()).__str__())
        print("Serial"+ raw.uuid)
        print("frequency"+ (raw.frequency).__str__())
        print('IS')
        print(raw.IS(binarize=binarize, threshold=threshold, freq=freq))
        print(raw.ISm(binarize=binarize, threshold=threshold))
        print(raw.ISp(binarize=binarize, threshold=threshold, period=period))
        print('IV')
        print(raw.IV(binarize=binarize, threshold=threshold, freq=freq))
        print(raw.IVm(binarize=binarize, threshold=threshold))
        print(raw.IVp(binarize=binarize, threshold=threshold, period=period))
        print('L5')
        print(raw.L5(binarize=binarize, threshold=threshold))
        print(raw.L5p(binarize=binarize, threshold=threshold, period=period))
        print('M10')
        print(raw.M10(binarize=binarize, threshold=threshold))
        print(raw.M10p(binarize=binarize, threshold=threshold,period=period))
        print('RA')
        print(raw.RA(binarize=binarize, threshold=threshold))
        print(raw.RAp(binarize=binarize, threshold=threshold,period=period))
        print(raw.pRA(threshold=threshold, start=None, period=period))
        print('pAR')
        print(raw.pAR(threshold=threshold, start=None, period=period))
        print('ADAT')
        print(raw.ADAT(binarize=binarize, threshold=threshold))
        print(raw.ADATp(binarize=binarize, threshold=threshold,period=period))
        print('k')
        print(raw.kRA(threshold=threshold, start=None, period=period))
        print(raw.kAR(threshold=threshold, start=None, period=period))
        tbl_res=[]
        pRA, pRA_weights = raw.pRA(threshold=threshold, start=None, period=period)
        temp_to_append = {'id': 1,
                          "Name": raw.name,
                          "Start_time": (raw.start_time).__str__(),
                          "Duration": (raw.duration()).__str__(),
                          "Serial": raw.uuid,
                          "frequency": (raw.frequency).__str__(),
                          "IS": raw.IS(binarize=binarize, threshold=threshold, freq=freq),
                          "ISm": raw.ISm(binarize=binarize, threshold=threshold),
                          'ISp':list(raw.ISp(binarize=binarize, threshold=threshold, period=period)),
                          "IV": raw.IV(binarize=binarize, threshold=threshold, freq=freq),
                          "IVm": raw.IVm(binarize=binarize, threshold=threshold),
                          'IVp': raw.IVp(binarize=binarize, threshold=threshold, period=period),
                          "L5": raw.L5(binarize=binarize, threshold=threshold),
                          "L5p": raw.L5p(binarize=binarize, threshold=threshold, period=period),
                          "M10": raw.M10(binarize=binarize, threshold=threshold),
                          "M10p": raw.M10p(binarize=binarize, threshold=threshold,period=period),
                          "RA": raw.RA(binarize=binarize, threshold=threshold),
                          "RAp": raw.RAp(binarize=binarize, threshold=threshold,period=period),
                          'pRA': pRA.to_list(),
                          'pAR': raw.pAR(threshold=threshold, start=None, period=period)[0].to_list(),
                          "ADAT": raw.ADAT(binarize=binarize, threshold=threshold),
                          "ADATp": raw.ADATp(binarize=binarize, threshold=threshold,period=period),
                          "kRA": raw.kRA(threshold=threshold, start=None, period=period),
                          "kAR": raw.kAR(threshold=threshold, start=None, period=period)
                          }
        tbl_res.append(temp_to_append)

        return JSONResponse(content={'status': 'Success', 'Result': tbl_res},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'Result': {}},
                            status_code=200)

@router.get("/cosinor_analysis_initial_values")
async def cosinoranalysisinitialvalues():
    test_status=''
    try:
        data = CosinorParameters.get_values(Cosinor())
        test_status = 'Unable to load Cosinor initial values.'
        return JSONResponse(content={'status':'Success',"cos_params": data},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, "cos_params": []}, status_code=200)


@router.get("/cosinor_analysis")
async def cosinoranalysis(workflow_id: str,
                          step_id: str,
                          run_id: str,
                          cosinor_parameters: str,
                          file:str):
    test_status = ''
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    try:
        test_status = 'Unable to retrieve actigraphy file.'
        # TODO : file or files?
        # raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv')
        raw = pyActigraphy.io.read_raw_rpx(path_to_storage + "/" + file)
        test_status = 'Unable to set cosinor values.'
        json_response = json.loads(cosinor_parameters)
        cosinor_obj = Cosinor()
        # set cosinor_obj values
        for elements in json_response:
            cosinor_obj = CosinorParameters.set_values_by_name(cosinor_obj,
                                           json_response[elements]['Name'],
                                           json_response[elements]['Value'],
                                           json_response[elements]['Vary'],
                                           json_response[elements]['Min'],
                                           json_response[elements]['Max'] if (json_response[elements]['Max'] != 'inf') else np.inf,
                                           json_response[elements]['Stderr'],
                                           json_response[elements]['Expr'],
                                           json_response[elements]['Brute_step'])

        # cosinor_obj.fit_initial_params.pretty_print()
        test_status = 'Unable to load Cosinor initial values.'
        data = CosinorParameters.get_values(cosinor_obj)
        test_status = 'Unable to execute Cosinor fit.'
        results = cosinor_obj.fit(raw, verbose=True)
        test_status = 'Unable to plot Cosinor fit.'
        fig = go.Figure(go.Scatter(x=raw.data.index.astype(str), y=raw.data))
        fig.write_image(path_to_storage + "/output/cosinor.svg", format="svg")
        test_status = 'Unable to create info file.'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data['results'] |= {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Kaplan Meier Fitter',
                "test_params": {
                    'file': file,
                    'cosinor_parameters': data
                },
                "test_results": {

                },
                "Output_datasets": [],
                'Saved_plots': [{"file": 'expertsystem/workflow/' + workflow_id + '/' + run_id + '/' +
                                             step_id + '/analysis_output/cosinor.svg'}
                                    ]
            }
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'Result': results.params.valuesdict(),
                'Akaike information criterium': results.aic,
                'Reduced Chi^2': results.redchi, "report": fit_report(results)},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'Result': {},
                'Akaike information criterium': {},
                'Reduced Chi^2': {}, "report": {}},
                            status_code=200)


class CosinorParameters(Cosinor):
    # def setinitialvalues(self):
    #
    #     self._Period_value = self.fit_initial_params['Period'].value
    #     self._Period_vary = self.fit_initial_params['Period'].vary
    #     self._Period_min = self.fit_initial_params['Period'].min
    #     self._Period_max = self.fit_initial_params['Period'].max
    #     self._Period_expr = self.fit_initial_params['Period'].expr
    #     self._Period_brute_step = self.fit_initial_params['Period'].brute_step
    #     self._Period_stderr = self.fit_initial_params['Period'].stderr
    #     self._Amplitude_value = self.fit_initial_params['Amplitude'].value
    #     self._Amplitude_vary = self.fit_initial_params['Amplitude'].vary
    #     self._Amplitude_min = self.fit_initial_params['Amplitude'].min
    #     self._Amplitude_max = self.fit_initial_params['Amplitude'].max
    #     self._Amplitude_expr = self.fit_initial_params['Amplitude'].expr
    #     self._Amplitude_brute_step = self.fit_initial_params['Amplitude'].brute_step
    #     self._Amplitude_stderr = self.fit_initial_params['Amplitude'].stderr
    #     self._Acrophase_value = self.fit_initial_params['Acrophase'].value
    #     self._Acrophase_vary = self.fit_initial_params['Acrophase'].vary
    #     self._Acrophase_min = self.fit_initial_params['Acrophase'].min
    #     self._Acrophase_max = self.fit_initial_params['Acrophase'].max
    #     self._Acrophase_expr = self.fit_initial_params['Acrophase'].expr
    #     self._Acrophase_brute_step = self.fit_initial_params['Acrophase'].brute_step
    #     self._Acrophase_stderr = self.fit_initial_params['Acrophase'].stderr
    #     self._Mesor_value = self.fit_initial_params['Mesor'].value
    #     self._Mesor_vary = self.fit_initial_params['Mesor'].vary
    #     self._Mesor_min = self.fit_initial_params['Mesor'].min
    #     self._Mesor_max = self.fit_initial_params['Mesor'].max
    #     self._Mesor_expr = self.fit_initial_params['Mesor'].expr
    #     self._Mesor_brute_step = self.fit_initial_params['Mesor'].brute_step
    #     self._Mesor_stderr = self.fit_initial_params['Mesor'].stderr
    #     return self
    # getter method
    def get_values(self):
        return [
            {
                'id': 1,
                'Name':'Period',
                'Value': self.fit_initial_params['Period'].value,
                'Vary': self.fit_initial_params['Period'].vary,
                'Min': self.fit_initial_params['Period'].min,
                'Max': self.fit_initial_params['Period'].max if not np.isinf(self.fit_initial_params['Period'].max) else 'inf',
                'Stderr':self.fit_initial_params['Period'].stderr,
                'Expr': self.fit_initial_params['Period'].expr,
                'Brute_step':self.fit_initial_params['Period'].brute_step
            },
            {
                'id': 2,
                'Name': 'Amplitude',
                'Value': self.fit_initial_params['Amplitude'].value,
                'Vary': self.fit_initial_params['Amplitude'].vary,
                'Min': self.fit_initial_params['Amplitude'].min,
                'Max': self.fit_initial_params['Amplitude'].max if not np.isinf(self.fit_initial_params['Amplitude'].max) else 'inf',
                'Stderr':self.fit_initial_params['Amplitude'].stderr,
                'Expr': self.fit_initial_params['Amplitude'].expr,
                'Brute_step': self.fit_initial_params['Amplitude'].brute_step
            },
            {
                'id': 3,
                'Name': 'Acrophase',
                'Value': self.fit_initial_params['Acrophase'].value,
                'Vary': self.fit_initial_params['Acrophase'].vary,
                'Min': self.fit_initial_params['Acrophase'].min,
                'Max': self.fit_initial_params['Acrophase'].max if not np.isinf(self.fit_initial_params['Acrophase'].max) else 'inf',
                'Stderr':self.fit_initial_params['Acrophase'].stderr,
                'Expr': self.fit_initial_params['Acrophase'].expr,
                'Brute_step': self.fit_initial_params['Acrophase'].brute_step
            },
            {
                'id': 4,
                'Name': 'Mesor',
                'Value': self.fit_initial_params['Mesor'].value,
                'Vary': self.fit_initial_params['Mesor'].vary,
                'Min': self.fit_initial_params['Mesor'].min,
                'Max': self.fit_initial_params['Mesor'].max if not np.isinf(self.fit_initial_params['Mesor'].max) else 'inf',
                'Stderr':self.fit_initial_params['Mesor'].stderr,
                'Expr': self.fit_initial_params['Mesor'].expr,
                'Brute_step': self.fit_initial_params['Mesor'].brute_step
            }
        ]

    # setter methods
    # def set_Mesor_max(self, x):
    #     self._Mesor_max = x

    def set_values_by_name(self,name:str, value:float, vary:bool, min:float, max:float, stderr:float, expr:str, brute_step:float):
        if name == 'Amplitude':
            self.fit_initial_params['Amplitude'].value = value
            self.fit_initial_params['Amplitude'].vary = vary
            self.fit_initial_params['Amplitude'].min = min
            self.fit_initial_params['Amplitude'].max = max
            self.fit_initial_params['Amplitude'].stderr = stderr
            self.fit_initial_params['Amplitude'].expr = expr
            self.fit_initial_params['Amplitude'].brute_step = brute_step
        elif name == 'Acrophase':
            self.fit_initial_params['Acrophase'].value = value
            self.fit_initial_params['Acrophase'].vary = vary
            self.fit_initial_params['Acrophase'].min = min
            self.fit_initial_params['Acrophase'].max = max
            self.fit_initial_params['Acrophase'].stderr = stderr
            self.fit_initial_params['Acrophase'].expr = expr
            self.fit_initial_params['Acrophase'].brute_step = brute_step
        elif name == 'Period':
            self.fit_initial_params['Period'].value = value
            self.fit_initial_params['Period'].vary = vary
            self.fit_initial_params['Period'].min = min
            self.fit_initial_params['Period'].max = max
            self.fit_initial_params['Period'].stderr = stderr
            self.fit_initial_params['Period'].expr = expr
            self.fit_initial_params['Period'].brute_step = brute_step
        elif name == 'Mesor':
            self.fit_initial_params['Mesor'].value = value
            self.fit_initial_params['Mesor'].vary = vary
            self.fit_initial_params['Mesor'].min = min
            self.fit_initial_params['Mesor'].max = max
            self.fit_initial_params['Mesor'].stderr = stderr
            self.fit_initial_params['Mesor'].expr = expr
            self.fit_initial_params['Mesor'].brute_step = brute_step
        return self
