import csv
from fastapi import Query, APIRouter
import pyActigraphy
import os
from datetime import datetime
from datetime import timedelta
import pandas as pd
from pyActigraphy.analysis import Cosinor
import plotly.graph_objects as go
from lmfit import fit_report


import pandas
import plotly.graph_objs as go
import plotly.io as pio
from app.utils.utils_general import get_local_storage_path, get_single_file_from_local_temp_storage, load_data_from_csv, \
    load_file_csv_direct

# import plotly.graph_objs as go

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
async def return_cole_kripke(workflow_id: str,
                             run_id: str,
                             step_id: str):
    raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv',
                                       start_time='2022-07-18 12:00:00',
                                       period='1 day',
                                       language='ENG_UK'
                                       )
    layout = go.Layout(title="Cole/Kripke Rest/Activity detection", xaxis=dict(title="Date time"),
                       yaxis=dict(title="Counts/period"), showlegend=False)
    CK = raw.CK()
    layout.update(yaxis2=dict(title='Classification', overlaying='y', side='right'), showlegend=True);
    output = go.Figure(data=[
        go.Scatter(x=raw.data.index.astype(str), y=raw.data, name='Data'),
        go.Scatter(x=CK.index.astype(str), y=CK, yaxis='y2', name='CK')
    ], layout=layout)
    pio.write_image(output, get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + 'ck_assessment.png')
    #output.show()


@router.get("/return_sadeh_scripp", tags=["actigraphy_analysis_assessment_algorithm"])
async def return_sadeh_scripp(workflow_id: str,
                              run_id: str,
                              step_id: str):
    raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv',
                                       start_time='2022-07-18 12:00:00',
                                       period='1 day',
                                       language='ENG_UK'
                                       )
    layout = go.Layout(title="Sadeh/Scripp Rest/Activity detection", xaxis=dict(title="Date time"),
                       yaxis=dict(title="Counts/period"), showlegend=False)
    sadeh = raw.Sadeh()
    scripps = raw.Scripps()
    layout.update(yaxis2=dict(title='Classification', overlaying='y', side='right'), showlegend=True);
    output = go.Figure(data=[
        go.Scatter(x=raw.data.index.astype(str), y=raw.data, name='Data'),
        go.Scatter(x=sadeh.index.astype(str), y=sadeh, yaxis='y2', name='Sadeh'),
        go.Scatter(x=scripps.index.astype(str), y=scripps, yaxis='y2', name='Scripps')
    ], layout=layout)
    pio.write_image(output,
                    get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + 'sadeh_scripp_assessment.png')
    #return output.show()


@router.get("/return_oakley", tags=["actigraphy_analysis_assessment_algorithm"])
async def return_oakley(workflow_id: str,
                        run_id: str,
                        step_id: str):
    raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv',
                                       start_time='2022-07-18 12:00:00',
                                       period='1 day',
                                       language='ENG_UK'
                                       )
    layout = go.Layout(title="Oakley Rest/Activity detection", xaxis=dict(title="Date time"),
                       yaxis=dict(title="Counts/period"), showlegend=False)
    oakley = raw.Oakley(threshold=40)
    oakley_auto = raw.Oakley(threshold='automatic')
    # sadeh = raw.Sadeh()
    # scripps = raw.Scripps()
    layout.update(yaxis2=dict(title='Classification', overlaying='y', side='right'), showlegend=True);
    output = go.Figure(data=[
        go.Scatter(x=raw.data.index.astype(str), y=raw.data, name='Data'),
        go.Scatter(x=oakley.index.astype(str), y=oakley, yaxis='y2', name='Oakley (thr: medium)'),
        go.Scatter(x=oakley_auto.index.astype(str), y=oakley_auto, yaxis='y2', name='Oakley (thr: automatic)')
    ], layout=layout)
    pio.write_image(output,
                    get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + 'oakley_assessment.png')
    #return output.show()


@router.get("/return_crespo", tags=["actigraphy_analysis_assessment_algorithm"])
async def return_crespo(workflow_id: str,
                        run_id: str,
                        step_id: str):
    raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv',
                                       start_time='2022-07-18 12:00:00',
                                       period='1 day',
                                       language='ENG_UK'
                                       )
    layout = go.Layout(title="Crespo Rest/Activity detection", xaxis=dict(title="Date time"),
                       yaxis=dict(title="Counts/period"), showlegend=False)
    crespo = raw.Crespo()
    crespo_6h = raw.Crespo(alpha='6h')
    crespo_zeta = raw.Crespo(estimate_zeta=True)
    # sadeh = raw.Sadeh()
    # scripps = raw.Scripps()
    layout.update(yaxis2=dict(title='Classification', overlaying='y', side='right'), showlegend=True);
    output = go.Figure(data=[
        go.Scatter(x=raw.data.index.astype(str), y=raw.data, name='Data'),
        go.Scatter(x=crespo.index.astype(str), y=crespo, yaxis='y2', name='Crespo'),
        go.Scatter(x=crespo_6h.index.astype(str), y=crespo_6h, yaxis='y2', name='Crespo (6h)'),
        go.Scatter(x=crespo_zeta.index.astype(str), y=crespo_zeta, yaxis='y2', name='Crespo (Automatic)')
    ], layout=layout)
    pio.write_image(output,
                    get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + 'crespo_assessment.png')
    #return output.show()


@router.get("/return_assessment_algorithm", tags=["actigraphy_analysis_assessment_algorithm"])
async def return_assessment_algorithm(workflow_id: str, run_id: str, step_id: str, algorithm: str):
    raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv',
                                       start_time='2022-07-18 12:00:00',
                                       period='1 day',
                                       language='ENG_UK'
                                       )
    # datetime_list = []
    # day_count = 1
    if (algorithm == "Cole - Kripke"):
        layout = go.Layout(title="Cole/Kripke Rest/Activity detection", xaxis=dict(title="Date time"),
                           yaxis=dict(title="Counts/period"), showlegend=False)
        CK = raw.CK()
        layout.update(yaxis2=dict(title='Classification', overlaying='y', side='right'), showlegend=True);
        output = go.Figure(data=[
            go.Scatter(x=raw.data.index.astype(str), y=raw.data, name='Data'),
            go.Scatter(x=CK.index.astype(str), y=CK, yaxis='y2', name='CK')
        ], layout=layout)
        pio.write_image(output, get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + 'assessment.png')
        # output.show()
    if (algorithm == "Sadeh - Scripp"):
        layout = go.Layout(title="Sadeh/Scripp Rest/Activity detection", xaxis=dict(title="Date time"),
                           yaxis=dict(title="Counts/period"), showlegend=False)
        sadeh = raw.Sadeh()
        scripps = raw.Scripps()
        layout.update(yaxis2=dict(title='Classification', overlaying='y', side='right'), showlegend=True);
        output = go.Figure(data=[
            go.Scatter(x=raw.data.index.astype(str), y=raw.data, name='Data'),
            go.Scatter(x=sadeh.index.astype(str), y=sadeh, yaxis='y2', name='Sadeh'),
            go.Scatter(x=scripps.index.astype(str), y=scripps, yaxis='y2', name='Scripps')
        ], layout=layout)
        pio.write_image(output,
                        get_local_storage_path(workflow_id, run_id,
                                               step_id) + "/output/" + 'assessment.png')
        # return output.show()
    if (algorithm == "Oakley"):
        layout = go.Layout(title="Oakley Rest/Activity detection", xaxis=dict(title="Date time"),
                           yaxis=dict(title="Counts/period"), showlegend=False)
        oakley = raw.Oakley(threshold=40)
        oakley_auto = raw.Oakley(threshold='automatic')
        # sadeh = raw.Sadeh()
        # scripps = raw.Scripps()
        layout.update(yaxis2=dict(title='Classification', overlaying='y', side='right'), showlegend=True);
        output = go.Figure(data=[
            go.Scatter(x=raw.data.index.astype(str), y=raw.data, name='Data'),
            go.Scatter(x=oakley.index.astype(str), y=oakley, yaxis='y2', name='Oakley (thr: medium)'),
            go.Scatter(x=oakley_auto.index.astype(str), y=oakley_auto, yaxis='y2', name='Oakley (thr: automatic)')
        ], layout=layout)
        pio.write_image(output,
                        get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + 'assessment.png')
        print(oakley)
        print(oakley_auto)
        # print(output.show())
    if (algorithm == "Crespo"):
        layout = go.Layout(title="Crespo Rest/Activity detection", xaxis=dict(title="Date time"),
                           yaxis=dict(title="Counts/period"), showlegend=False)
        crespo = raw.Crespo()
        crespo_6h = raw.Crespo(alpha='6h')
        crespo_zeta = raw.Crespo(estimate_zeta=True)
        # sadeh = raw.Sadeh()
        # scripps = raw.Scripps()
        layout.update(yaxis2=dict(title='Classification', overlaying='y', side='right'), showlegend=True);
        output = go.Figure(data=[
            go.Scatter(x=raw.data.index.astype(str), y=raw.data, name='Data'),
            go.Scatter(x=crespo.index.astype(str), y=crespo, yaxis='y2', name='Crespo'),
            go.Scatter(x=crespo_6h.index.astype(str), y=crespo_6h, yaxis='y2', name='Crespo (6h)'),
            go.Scatter(x=crespo_zeta.index.astype(str), y=crespo_zeta, yaxis='y2', name='Crespo (Automatic)')
        ], layout=layout)
        pio.write_image(output,
                        get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + 'assessment.png')
        # return output.show()


@router.get("/return_weekly_activity", tags=["actigraphy_analysis"])
async def return_weekly_activity(workflow_id: str,
                                 run_id: str,
                                 step_id: str,
                                 start_date: str,
                                 end_date: str):
    # Convert a String to a Date in Python
    # Date and time in format "YYYY/MM/DD hh:mm:ss"
    format_string = "%Y/%m/%d %H:%M:%S"

    # Convert start date string to date using strptime
    start_date_dt = datetime.strptime(start_date, format_string).date()
    # Convert end date string to date using strptime
    end_date_dt = datetime.strptime(end_date, format_string).date()

    datetime_list = []
    for i in range(0, (end_date_dt - start_date_dt).days + 1):
        datetime_list.append(  str(start_date_dt + timedelta(days=i)) + str(" 12:00:00")  ) #<-- here
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
        # output.write_image("/output/actigraphy_visualisation.svg")
        # export as static image
        # pio.write_image(output, "C://neurodesktop-storage//runtime_config//workflow_1//run_1//step_1//output//" + str(day_count) + "actigraphy_visualisation.png")
        pio.write_image(output, get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + str(
            day_count) + '_actigraphy_visualisation.png')
        day_count = day_count + 1
        # return output
        #output.show()

    # raw.name
    # raw.start_time
    # raw.duration()
    # raw.uuid
    # raw.frequency


@router.get("/rest_to_activity_probability", tags=["actigraphy_analysis"])
async def rest_to_activity_probability():
    raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv',
                                       period='7days')

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

def return_rawObject():
    xx = 0
    try:
        # fpath = f"C:/Users/gdoukas/PycharmProjects/data-analytics-and-visualisation-backend/example_data/actigraph/example_data/actigraph/Neurophy_Actigraph.csv"
        raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/test_sample.csv', 'FR')
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
                            i: int,
                            binarize: bool | None = Query(default=True),
                            threshold: int | None = Query(default=4),
                            metric: str | None = Query(default='IS', enum=['IS','ISm','IV','IVm', 'L5','M10','RA','L5p','ADAT','ADATp','M10p','RAp','kRA','kAR']),
                            prefix_offset: str | None = Query(default="Hour",
                                                              regex="^(Hour)$|^(Minute)$|^(Day)$")):

    raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv')
    if metric == 'IS':
        if prefix_offset == 'Hour':
            freq = f"{i}{pd.offsets.Hour._prefix}"
        else:
            freq = f"{i}{pd.offsets.Minute._prefix}"
        xx = raw.IS(binarize=binarize, threshold=threshold, freq=freq)

        return {'IS': xx}
    elif metric == 'ISm':
        xx = raw.ISm(binarize=binarize, threshold=threshold)
        return {'ISm': xx}
    elif metric == 'IV':
        if prefix_offset == 'Hour':
            freq = f"{i}{pd.offsets.Hour._prefix}"
        else:
            freq = f"{i}{pd.offsets.Minute._prefix}"
        xx = raw.IV(binarize=binarize, threshold=threshold, freq=freq)

        return {'IV': xx}
    elif metric == 'IVm':
        xx = raw.IVm(binarize=binarize, threshold=threshold)
        return {'IVm': xx}
    elif metric == 'L5':
        xx = raw.L5(binarize=binarize, threshold=threshold)
        return {'L5': xx}
    elif metric == 'M10':
        xx = raw.M10(binarize=binarize, threshold=threshold)
        return {'M10': xx}
    elif metric == 'RA':
        xx = raw.RA(binarize=binarize, threshold=threshold)
        return {'RA': xx}
    elif metric == 'L5p':
        if prefix_offset == 'Hour':
            freq = f"{i}{pd.offsets.Hour._prefix}"
        elif prefix_offset == 'Minute':
            freq = f"{i}{pd.offsets.Minute._prefix}"
        else:
            freq = f"{i}{pd.offsets.Day._prefix}"
        xx = raw.L5p(binarize=binarize, threshold=threshold, period=freq)
        return {'L5p': xx}
    elif metric == 'ADAT':
        xx = raw.ADAT(binarize=binarize, threshold=threshold)
        return {'ADAT': xx}
    elif metric == 'ADATp':
        if prefix_offset == 'Hour':
            freq = f"{i}{pd.offsets.Hour._prefix}"
        elif prefix_offset == 'Minute':
            freq = f"{i}{pd.offsets.Minute._prefix}"
        else:
            freq = f"{i}{pd.offsets.Day._prefix}"
        xx = raw.ADATp(binarize=binarize, threshold=threshold,period=freq)
        return {'ADATp': xx}
    elif metric == 'M10p':
        if prefix_offset == 'Hour':
            freq = f"{i}{pd.offsets.Hour._prefix}"
        elif prefix_offset == 'Minute':
            freq = f"{i}{pd.offsets.Minute._prefix}"
        else:
            freq = f"{i}{pd.offsets.Day._prefix}"
        xx = raw.M10p(binarize=binarize, threshold=threshold,period=freq)
        return {'M10p': xx}
    elif metric == 'RAp':
        if prefix_offset == 'Hour':
            freq = f"{i}{pd.offsets.Hour._prefix}"
        elif prefix_offset == 'Minute':
            freq = f"{i}{pd.offsets.Minute._prefix}"
        else:
            freq = f"{i}{pd.offsets.Day._prefix}"
        xx = raw.RAp(binarize=binarize, threshold=threshold,period=freq)
        return {'RAp': xx}
    elif metric == 'kRA':
        xx = raw.kRA()
        return {'kRA': xx}
    elif metric == 'kAR':
        xx = raw.kAR()
        return {'kAR': xx}

@router.get("/cosinor_analysis_initial_values")
async def cosinoranalysisinitialvalues(workflow_id: str,
                                       step_id: str,
                                       run_id: str):

    cosinor = Cosinor()

    return {'Initial Values': cosinor.fit_initial_params.valuesdict()}

@router.get("/cosinor_analysis")
async def cosinoranalysis(workflow_id: str,
                          step_id: str,
                          run_id: str,
                          Period: int | None = Query(default=None),
                          set_new_values: bool | None = Query(default=False),
                          change_period: bool | None = Query(default=False)):

    raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv')

    cosinor = Cosinor()

    if set_new_values == False:
        results = cosinor.fit(raw, verbose=True)
        return {'Result':results.params.valuesdict(), 'Akaike information criterium': results.aic, 'Reduced Chi^2': results.redchi, "report":fit_report(results)}
    else:
        if change_period == True:
            cosinor.fit_initial_params['Period'].value = Period
        results = cosinor.fit(raw, verbose=True)
        # TODO: create plot
        return {'Result': results.params.valuesdict(), 'Akaike information criterium': results.aic,
                'Reduced Chi^2': results.redchi, "report": fit_report(results)}









