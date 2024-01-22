import csv
import inspect
from time import strftime
import json
import numpy as np
from fastapi import Query, APIRouter
import pyActigraphy
import os
import datetime
from datetime import datetime
from datetime import timedelta
import pandas as pd
from pyActigraphy.analysis import Cosinor
# from pyActigraphy.io.rpx.rpx import language
import plotly.graph_objects as go
from lmfit import fit_report
import plotly
from pyActigraphy.analysis import FLM
from pyActigraphy.analysis import SSA
# The DFA methods are part of the Fractal module:
from pyActigraphy.analysis import Fractal
import shutil
from datetime import datetime
import pandas
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
from app.utils.utils_general import get_local_storage_path, get_single_file_from_local_temp_storage, load_data_from_csv, \
    load_file_csv_direct

# import plotly.graph_objs as go
from starlette.responses import JSONResponse

from app.utils.utils_general import get_local_storage_path

router = APIRouter()

@router.get("/return_dates", tags=["actigraphy_analysis"])
async def return_dates(workflow_id: str,
                      run_id: str,
                      step_id: str,
                      dataset: str):
    # Import dataset as pd dataframe excluding the first 150 rows
    json_dataframe = ''
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    df = pd.read_csv(path_to_storage + '/' + dataset, skiprows=150)
    df.drop(df.columns[[12]], axis=1, inplace=True)
    df["DateTime"] = df[["Date", "Time"]].agg(" ".join, axis=1)
    mylist = df['DateTime'].tolist()
    new_list = []
    numbers_list = []

    for item in mylist:
        if item[:2] not in numbers_list:
            new_list.append(item[:10])
            numbers_list.append(item[:2])

    # Convert a String to a Date in Python
    # Date and time in format "YYYY/MM/DD hh:mm:ss"
    format_string = "%d/%m/%Y"

    final_dates_list = []

    for item in new_list:
        # Convert start date string to date using strptime
        datetime_date = datetime.strptime(item, format_string).date()
        final_dates_list.append(str(datetime_date).replace("-", "/") + str(" 12:00:00"))

    return{'dates': list(final_dates_list)}

@router.get("/return_cole_kripke", tags=["actigraphy_analysis_assessment_algorithm"])
async def return_cole_kripke(workflow_id: str,
                             run_id: str,
                             step_id: str):
    raw = pyActigraphy.io.read_raw_rpx('/neurodesktop-storage/runtime_config/workflow_3fa85f64-5717-4562-b3fc-2c963f66afa6/run_3fa85f64-5717-4562-b3fc-2c963f66afa6/step_3fa85f64-5717-4562-b3fc-2c963f66afa6/0345-024_18_07_2022_13_00_00_New_Analysis.csv',
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
    graphJSON = plotly.io.to_json(output, pretty=True)
    # print(graphJSON)
    return {"figure": graphJSON}
    #output.show()

@router.get("/return_daily_activity", tags=["actigraphy_analysis"])
async def return_daily_activity(workflow_id: str,
                                 run_id: str,
                                 step_id: str,
                                 dataset: str,
                                 algorithm: str,
                                 start_date: str,
                                 end_date: str):
    # Convert a String to a Date in Python
    # Date and time in format "YYYY/MM/DD hh:mm:ss"
    format_string = "%Y/%m/%d %H:%M:%S"

    # Convert start date string to date using strptime
    start_date_dt = datetime.strptime(start_date, format_string).date()
    # Convert end date string to date using strptime
    end_date_dt = datetime.strptime(end_date, format_string).date()
    graph_JSON_list = []
    graphJSON = ""
    datetime_list = []
    for i in range(0, (end_date_dt - start_date_dt).days + 1):
        datetime_list.append(str(start_date_dt + timedelta(days=i)) + str(" 12:00:00"))  # <-- here
    day_count = 1
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    for i in datetime_list[:-1]:
        raw = pyActigraphy.io.read_raw_rpx(
            path_to_storage + '/' + dataset,
            start_time=i,
            period='1 day',
            language='ENG_UK'
        )
        if (algorithm == "Cole - Kripke"):
            CK = raw.CK()
            # print(raw.data.index)
            layout = go.Layout(
                title="Cole/Kripke Rest/Activity detection",
                xaxis=dict(title="Date time"),
                yaxis=dict(title="Counts/period"),
                showlegend=False
            )
            fig = make_subplots(rows=2, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.1)
            fig.add_scatter(x=raw.data.index.astype(str), y=raw.data, mode='markers', name='Visualisation', row=1, col=1)
            fig.add_scatter(x=CK.index.astype(str), y=CK, mode='lines', name='Cole/Kripke Assessment', row=2, col=1)
            output = go.Figure(data=[go.Scatter(x=raw.data.index.astype(str), y=raw.data, mode='markers')], layout=layout)
            fig.update_layout(title='Cole/Kripke Rest/Activity detection day ' + str(day_count), width=800, height=800, dragmode='select',
                              activeselection=dict(fillcolor='yellow'))
            graphJSON = plotly.io.to_json(fig, pretty=True)
            graph_JSON_list.append(plotly.io.to_json(fig, pretty=True))
            # fig.show()
            day_count = day_count + 1
            # return {"figure": graphJSON}
        if (algorithm == "Sadeh - Scripp"):
            sadeh = raw.Sadeh()
            scripps = raw.Scripps()
            layout = go.Layout(title="Sadeh/Scripp Rest/Activity detection",
                               xaxis=dict(title="Date time"),
                               yaxis=dict(title="Counts/period"),
                               showlegend=False)
            fig = make_subplots(rows=2, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.1)
            fig.add_scatter(x=raw.data.index.astype(str), y=raw.data, mode='markers', name='Visualisation', row=1, col=1)
            fig.add_scatter(x=sadeh.index.astype(str), y=sadeh, mode='lines', name='Sadeh Assessment', row=2, col=1)
            fig.add_scatter(x=scripps.index.astype(str), y=scripps, mode='lines', name='Scripps Assessment', row=2, col=1)
            output = go.Figure(data=[go.Scatter(x=raw.data.index.astype(str), y=raw.data, mode='markers')], layout=layout)
            fig.update_layout(title='Sadeh/Scripp Rest/Activity detection day ' + str(day_count), width=800, height=800, dragmode='select',
                              activeselection=dict(fillcolor='yellow'))
            graphJSON = plotly.io.to_json(fig, pretty=True)
            graph_JSON_list.append(plotly.io.to_json(fig, pretty=True))
            day_count = day_count + 1
            # return {"figure": graphJSON}
        if (algorithm == "Oakley"):
            oakley = raw.Oakley(threshold=40)
            oakley_auto = raw.Oakley(threshold='automatic')
            layout = go.Layout(title="Oakley Rest/Activity detection",
                               xaxis=dict(title="Date time"),
                               yaxis=dict(title="Counts/period"),
                               showlegend=False)
            fig = make_subplots(rows=2, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.1)
            fig.add_scatter(x=raw.data.index.astype(str), y=raw.data, mode='markers', name='Visualisation', row=1, col=1)
            fig.add_scatter(x=oakley.index.astype(str), y=oakley, mode='lines', name='Oakley (thr: medium)', row=2, col=1)
            fig.add_scatter(x=oakley_auto.index.astype(str), y=oakley_auto, mode='lines', name='Oakley (thr: automatic)', row=2, col=1)
            fig.update_layout(title='Oakley Rest/Activity detection day ' + str(day_count), width=800, height=800, dragmode='select',
                              activeselection=dict(fillcolor='yellow'))
            graphJSON = plotly.io.to_json(fig, pretty=True)
            graph_JSON_list.append(plotly.io.to_json(fig, pretty=True))
            day_count = day_count + 1
            # return {"figure": graphJSON}
        if (algorithm == "Crespo"):
            crespo = raw.Crespo()
            crespo_6h = raw.Crespo(alpha='6h')
            crespo_zeta = raw.Crespo(estimate_zeta=True)
            layout = go.Layout(title="Oakley Rest/Activity detection",
                               xaxis=dict(title="Date time"),
                               yaxis=dict(title="Counts/period"),
                               showlegend=False)

            fig = make_subplots(rows=2, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.1)
            fig.add_scatter(x=raw.data.index.astype(str), y=raw.data, mode='markers', name='Visualisation', row=1, col=1)
            fig.add_scatter(x=crespo.index.astype(str), y=crespo, mode='lines', name='Crespo', row=2, col=1)
            fig.add_scatter(x=crespo_6h.index.astype(str), y=crespo_6h, mode='lines', name='Crespo (6h)',
                            row=2, col=1)
            fig.add_scatter(x=crespo_zeta.index.astype(str), y=crespo_zeta, mode='lines', name='Crespo (Automatic)',
                            row=2, col=1)
            fig.update_layout(title='Oakley Rest/Activity detection day ' + str(day_count), width=800, height=800, dragmode='select',
                              activeselection=dict(fillcolor='yellow'))
            graph_JSON_list.append(plotly.io.to_json(fig, pretty=True))
            # graphJSON = graphJSON + plotly.io.to_json(fig, pretty=True)
            day_count = day_count + 1
            # print(graphJSON)
            # print(day_count)
            # print(datetime_list)
            # fig.show()
            # print(graph_JSON_list)
    return {"figure": graph_JSON_list}

    # output.show()

def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

@router.get("/return_daily_activity_activity_status_area", tags=["actigraphy_analysis"])
async def return_daily_activity_activity_status_area(workflow_id: str,
                                                     run_id: str,
                                                     step_id: str,
                                                     dataset: str,
                                                     start_date: str,
                                                     end_date: str):
    # Convert a String to a Date in Python
    # Date and time in format "YYYY/MM/DD hh:mm:ss"
    format_string = "%Y/%m/%d %H:%M:%S"
    print(start_date)
    print(end_date)
    # Convert start date string to date using strptime
    start_date_dt = datetime.strptime(start_date, format_string).date()
    # Convert end date string to date using strptime
    end_date_dt = datetime.strptime(end_date, format_string).date()
    graph_JSON_list = []
    graphJSON = ""
    datetime_list = []
    for i in range(0, (end_date_dt - start_date_dt).days + 1):
        datetime_list.append(str(start_date_dt + timedelta(days=i)) + str(" 12:00:00"))  # <-- here
    #     print(datetime_list)
    day_count = 1
    x0_count = 0
    x1_count = 0
    cols = plotly.colors.DEFAULT_PLOTLY_COLORS

    mylist = []
    for i in datetime_list:
        new = i.replace("-", "/")
        mylist.append(new)
    # print(mylist)
    date_list_obj = []
    for date in mylist:
        date_object = datetime.strptime(date, '%Y/%m/%d %H:%M:%S')
        # print(date_object)
        date_list_obj.append(date_object)
    date_list = [date_obj.strftime('%d/%m/%Y %H:%M:%S') for date_obj in date_list_obj]
    # print(date_list)

    # create a datetime list to compare with datetimes from df
    start = datetime.strptime(date_list[0], '%d/%m/%Y %H:%M:%S')
    end = datetime.strptime(date_list[-1], '%d/%m/%Y %H:%M:%S')
    # print(start)
    # print(end)

    dts = [dt.strftime('%d/%m/%Y %.H:%M:%S') for dt in
           datetime_range(start, end,
                          timedelta(seconds=15))]
    #     print(dts)

    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)

    df = pd.read_csv(path_to_storage + '/' + dataset, skiprows=150)
    df["Datetime"] = df[["Date", "Time"]].apply(lambda x: " ".join(x), axis=1)
    df.drop(df.columns[[12]], axis=1, inplace=True)
    #     for datetime in date_list:
    df = df.drop(index=[row for row in df.index if df.loc[row, 'Datetime'] not in dts])
    #     display(df)

    df["Prev_Interval_Status"] = df["Interval Status"].shift(+1)
    df["Next_Interval_Status"] = df["Interval Status"].shift(-1)
    df['Flag_1'] = np.where((df['Interval Status'] == "REST-S") & (df['Prev_Interval_Status'] != "REST-S"), "Slept", "")
    df['Flag_2'] = np.where((df['Interval Status'] == "REST-S") & (df['Next_Interval_Status'] != "REST-S"), "Woke", "")
    #     print(df.to_string())
    # print(df)

    df_x0_list = df
    df_x0_list = df_x0_list.drop(index=[row for row in df_x0_list.index if "Slept" != df_x0_list.loc[row, 'Flag_1']])
    print(df_x0_list)
    df_x1_list = df
    df_x1_list = df_x1_list.drop(index=[row for row in df_x1_list.index if "Woke" != df_x1_list.loc[row, 'Flag_2']])
    print(df_x1_list)

    x0 = []
    # for index, row in df_x0_list.iterrows():
    #     if df_x0_list['Date'].eq(df_x0_list['Date'].shift(-1)):
    #     x0 = []
    x0 = df_x0_list['Datetime'].tolist()
    #     x0.pop()
    print(x0)
    x1 = []
    x1 = df_x1_list['Datetime'].tolist()
    #     x1.pop()
    print(x1)

    x_list = [list(item) for item in zip(x0, x1)]
    print(x_list)

    # Manipulate x_list
    x_list_fixed = []
    for sublist in x_list:
        help_list = []
        for item in sublist:
            new_format_x0 = item.replace("/", "-")
            date_object = datetime.strptime(new_format_x0, '%d-%m-%Y %H:%M:%S')
            date_string = date_object.strftime('%Y-%m-%d %H:%M:%S')
            help_list.append(date_string)
        x_list_fixed.append(help_list)
    print(x_list_fixed)

    # Manipulate x0_list
    x0_list = []
    for i in x0:
        new_format_x0 = i.replace("/", "-")
        date_object = datetime.strptime(new_format_x0, '%d-%m-%Y %H:%M:%S')
        date_string = date_object.strftime('%Y-%m-%d %H:%M:%S')
        x0_list.append(date_string)

    # Manipulate x1_list
    x1_list = []
    for i in x1:
        new_format_x1 = i.replace("/", "-")
        date_object = datetime.strptime(new_format_x1, '%d-%m-%Y %H:%M:%S')
        date_string = date_object.strftime('%Y-%m-%d %H:%M:%S')
        x1_list.append(date_string)
    # print(x1_list)

    for i in range(len(datetime_list) - 1):
        fig = make_subplots(rows=i + 1, cols=1,
                            # shared_yaxes=True,
                            vertical_spacing=0.05,
                            subplot_titles=("Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6"))

    raw_start_time = 0

    for sublist in x_list_fixed:
        raw = pyActigraphy.io.read_raw_rpx(
            path_to_storage + '/' + dataset,
            start_time=datetime_list[raw_start_time],
            period='1 day',
            language='ENG_UK'
        )
        for item in sublist:
            first_date = sublist[0][:10]
            second_date = sublist[1][:10]
        if (first_date == second_date):
            print("yes")
            fig.add_scatter(x=raw.data.index.astype(str), y=raw.data, mode='lines', line=dict(width=2, color=cols[0]),
                            name='Visualisation', row=day_count, col=1)
            fig.add_vline(x=sublist[0], row=day_count, col=1)
            fig.add_vline(x=sublist[1], row=day_count, col=1)
            fig.add_vrect(x0=sublist[0], x1=sublist[1],
                          annotation_text="REST Period", annotation_position="top left",
                          fillcolor="green", opacity=0.25, line_width=0, row=day_count, col=1)
            x0_count = x0_count + 1
            x1_count = x1_count + 1
            # fig.add_scatter(x=raw.mask.index.astype(str), y=raw.mask, yaxis='y2',mode='lines', line=dict(width=2, color=cols[1]), name='Inactivity Mask', row=day_count, col=1)
            fig.update_layout(title='Activity Visualisation', width=800, height=1400,
                              dragmode='select',
                              activeselection=dict(fillcolor='yellow'))
        else:
            print("no")
            fig.add_scatter(x=raw.data.index.astype(str), y=raw.data, mode='lines', line=dict(width=2, color=cols[0]),
                            name='Visualisation', row=day_count, col=1)
            fig.add_vline(x=sublist[0], row=day_count, col=1)
            fig.add_vline(x=sublist[1], row=day_count, col=1)
            fig.add_vrect(x0=sublist[0], x1=sublist[1],
                          annotation_text="REST Period", annotation_position="top left",
                          fillcolor="green", opacity=0.25, line_width=0, row=day_count, col=1)
            x0_count = x0_count + 1
            x1_count = x1_count + 1
            # fig.add_scatter(x=raw.mask.index.astype(str), y=raw.mask, yaxis='y2',mode='lines', line=dict(width=2, color=cols[1]), name='Inactivity Mask', row=day_count, col=1)
            fig.update_layout(title='Activity Visualisation', width=800, height=1400,
                              dragmode='select',
                              activeselection=dict(fillcolor='yellow'))
            day_count = day_count + 1
            raw_start_time = raw_start_time + 1

    # for i in datetime_list[:-2]:
    #     raw = pyActigraphy.io.read_raw_rpx(
    #         'copy.csv',
    #         start_time=i,
    #         period='1 day',
    #         language='ENG_UK'
    #     )
    #     # raw.create_inactivity_mask(duration='2h00min')
    #     fig.add_scatter(x=raw.data.index.astype(str), y=raw.data, mode='lines', line=dict(width=2, color=cols[0]),
    #                     name='Visualisation', row=day_count, col=1)
    #     fig.add_vline(x=x0_list[x0_count], row=day_count, col=1)
    #     fig.add_vline(x=x1_list[x1_count], row=day_count, col=1)
    #     fig.add_vrect(x0=x0_list[x0_count], x1=x1_list[x1_count],
    #                   annotation_text="REST Period", annotation_position="top left",
    #                   fillcolor="green", opacity=0.25, line_width=0, row=day_count, col=1)
    #     x0_count = x0_count + 1
    #     x1_count = x1_count + 1
    #     # fig.add_scatter(x=raw.mask.index.astype(str), y=raw.mask, yaxis='y2',mode='lines', line=dict(width=2, color=cols[1]), name='Inactivity Mask', row=day_count, col=1)
    #     fig.update_layout(title='Activity Visualisation', width=800, height=1400,
    #                       dragmode='select',
    #                       activeselection=dict(fillcolor='yellow'))
    #     day_count = day_count + 1

    # fig.show()
    # print(datetime_list)
    graphJSON = plotly.io.to_json(fig, pretty=True)
    return {"visualisation_figure": graphJSON}

@router.get("/return_final_daily_activity_activity_status_area", tags=["actigraphy_analysis"])
async def return_final_daily_activity_activity_status_area(workflow_id: str,
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
    graph_JSON_list = []
    graphJSON = ""
    datetime_list = []
    for i in range(0, (end_date_dt - start_date_dt).days + 1):
        datetime_list.append(str(start_date_dt + timedelta(days=i)) + str(" 12:00:00"))  # <-- here
    #     print(datetime_list)
    day_count = 1
    x0_count = 0
    x1_count = 0
    cols = plotly.colors.DEFAULT_PLOTLY_COLORS

    mylist = []
    for i in datetime_list:
        new = i.replace("-", "/")
        mylist.append(new)
    # print(mylist)
    date_list_obj = []
    for date in mylist:
        date_object = datetime.strptime(date, '%Y/%m/%d %H:%M:%S')
        # print(date_object)
        date_list_obj.append(date_object)
    date_list = [date_obj.strftime('%d/%m/%Y %H:%M:%S') for date_obj in date_list_obj]
    # print(date_list)

    # create a datetime list to compare with datetimes from df
    start = datetime.strptime(date_list[0], '%d/%m/%Y %H:%M:%S')
    end = datetime.strptime(date_list[-1], '%d/%m/%Y %H:%M:%S')
    # print(start)
    # print(end)

    dts = [dt.strftime('%d/%m/%Y %#H:%M:%S') for dt in
           datetime_range(start, end,
                          timedelta(seconds=15))]
    #     print(dts)

    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)

    df = pd.read_csv(path_to_storage + '/output/' + 'NewAnalysisCopy.csv', skiprows=150)
    df["Datetime"] = df[["Date", "Time"]].apply(lambda x: " ".join(x), axis=1)
    df.drop(df.columns[[12]], axis=1, inplace=True)
    #     for datetime in date_list:
    df = df.drop(index=[row for row in df.index if df.loc[row, 'Datetime'] not in dts])
    #     display(df)

    df["Prev_Interval_Status"] = df["Interval Status"].shift(+1)
    df["Next_Interval_Status"] = df["Interval Status"].shift(-1)
    df['Flag_1'] = np.where((df['Interval Status'] == "REST-S") & (df['Prev_Interval_Status'] != "REST-S"), "Slept", "")
    df['Flag_2'] = np.where((df['Interval Status'] == "REST-S") & (df['Next_Interval_Status'] != "REST-S"), "Woke", "")
    #     print(df.to_string())
    # print(df)

    df_x0_list = df
    df_x0_list = df_x0_list.drop(index=[row for row in df_x0_list.index if "Slept" != df_x0_list.loc[row, 'Flag_1']])
    print(df_x0_list)
    df_x1_list = df
    df_x1_list = df_x1_list.drop(index=[row for row in df_x1_list.index if "Woke" != df_x1_list.loc[row, 'Flag_2']])
    print(df_x1_list)

    x0 = []
    # for index, row in df_x0_list.iterrows():
    #     if df_x0_list['Date'].eq(df_x0_list['Date'].shift(-1)):
    #     x0 = []
    x0 = df_x0_list['Datetime'].tolist()
    #     x0.pop()
    print(x0)
    x1 = []
    x1 = df_x1_list['Datetime'].tolist()
    #     x1.pop()
    print(x1)

    x_list = [list(item) for item in zip(x0, x1)]
    print(x_list)

    # Manipulate x_list
    x_list_fixed = []
    for sublist in x_list:
        help_list = []
        for item in sublist:
            new_format_x0 = item.replace("/", "-")
            date_object = datetime.strptime(new_format_x0, '%d-%m-%Y %H:%M:%S')
            date_string = date_object.strftime('%Y-%m-%d %H:%M:%S')
            help_list.append(date_string)
        x_list_fixed.append(help_list)
    print(x_list_fixed)

    # Manipulate x0_list
    x0_list = []
    for i in x0:
        new_format_x0 = i.replace("/", "-")
        date_object = datetime.strptime(new_format_x0, '%d-%m-%Y %H:%M:%S')
        date_string = date_object.strftime('%Y-%m-%d %H:%M:%S')
        x0_list.append(date_string)

    # Manipulate x1_list
    x1_list = []
    for i in x1:
        new_format_x1 = i.replace("/", "-")
        date_object = datetime.strptime(new_format_x1, '%d-%m-%Y %H:%M:%S')
        date_string = date_object.strftime('%Y-%m-%d %H:%M:%S')
        x1_list.append(date_string)
    # print(x1_list)

    for i in range(len(datetime_list) - 1):
        fig = make_subplots(rows=i + 1, cols=1,
                            # shared_yaxes=True,
                            vertical_spacing=0.05,
                            subplot_titles=("Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6"))

    raw_start_time = 0

    for sublist in x_list_fixed:
        raw = pyActigraphy.io.read_raw_rpx(
            path_to_storage + '/output/' + 'NewAnalysisCopy.csv',
            start_time=datetime_list[raw_start_time],
            period='1 day',
            language='ENG_UK'
        )
        for item in sublist:
            first_date = sublist[0][:10]
            second_date = sublist[1][:10]
        if (first_date == second_date):
            print("yes")
            fig.add_scatter(x=raw.data.index.astype(str), y=raw.data, mode='lines', line=dict(width=2, color=cols[0]),
                            name='Visualisation', row=day_count, col=1)
            fig.add_vline(x=sublist[0], row=day_count, col=1)
            fig.add_vline(x=sublist[1], row=day_count, col=1)
            fig.add_vrect(x0=sublist[0], x1=sublist[1],
                          annotation_text="REST Period", annotation_position="top left",
                          fillcolor="green", opacity=0.25, line_width=0, row=day_count, col=1)
            x0_count = x0_count + 1
            x1_count = x1_count + 1
            # fig.add_scatter(x=raw.mask.index.astype(str), y=raw.mask, yaxis='y2',mode='lines', line=dict(width=2, color=cols[1]), name='Inactivity Mask', row=day_count, col=1)
            fig.update_layout(title='Activity Visualisation', width=800, height=1400,
                              dragmode='select',
                              activeselection=dict(fillcolor='yellow'))
        else:
            print("no")
            fig.add_scatter(x=raw.data.index.astype(str), y=raw.data, mode='lines', line=dict(width=2, color=cols[0]),
                            name='Visualisation', row=day_count, col=1)
            fig.add_vline(x=sublist[0], row=day_count, col=1)
            fig.add_vline(x=sublist[1], row=day_count, col=1)
            fig.add_vrect(x0=sublist[0], x1=sublist[1],
                          annotation_text="REST Period", annotation_position="top left",
                          fillcolor="green", opacity=0.25, line_width=0, row=day_count, col=1)
            x0_count = x0_count + 1
            x1_count = x1_count + 1
            # fig.add_scatter(x=raw.mask.index.astype(str), y=raw.mask, yaxis='y2',mode='lines', line=dict(width=2, color=cols[1]), name='Inactivity Mask', row=day_count, col=1)
            fig.update_layout(title='Activity Visualisation', width=800, height=1400,
                              dragmode='select',
                              activeselection=dict(fillcolor='yellow'))
            day_count = day_count + 1
            raw_start_time = raw_start_time + 1

    # for i in datetime_list[:-2]:
    #     raw = pyActigraphy.io.read_raw_rpx(
    #         'copy.csv',
    #         start_time=i,
    #         period='1 day',
    #         language='ENG_UK'
    #     )
    #     # raw.create_inactivity_mask(duration='2h00min')
    #     fig.add_scatter(x=raw.data.index.astype(str), y=raw.data, mode='lines', line=dict(width=2, color=cols[0]),
    #                     name='Visualisation', row=day_count, col=1)
    #     fig.add_vline(x=x0_list[x0_count], row=day_count, col=1)
    #     fig.add_vline(x=x1_list[x1_count], row=day_count, col=1)
    #     fig.add_vrect(x0=x0_list[x0_count], x1=x1_list[x1_count],
    #                   annotation_text="REST Period", annotation_position="top left",
    #                   fillcolor="green", opacity=0.25, line_width=0, row=day_count, col=1)
    #     x0_count = x0_count + 1
    #     x1_count = x1_count + 1
    #     # fig.add_scatter(x=raw.mask.index.astype(str), y=raw.mask, yaxis='y2',mode='lines', line=dict(width=2, color=cols[1]), name='Inactivity Mask', row=day_count, col=1)
    #     fig.update_layout(title='Activity Visualisation', width=800, height=1400,
    #                       dragmode='select',
    #                       activeselection=dict(fillcolor='yellow'))
    #     day_count = day_count + 1

    # fig.show()
    # print(datetime_list)
    graphJSON = plotly.io.to_json(fig, pretty=True)
    return {"visualisation_figure_final": graphJSON}

@router.get("/return_initial_dataset", tags=["actigraphy_analysis"])
async def return_initial_dataset(workflow_id: str,
                                 run_id: str,
                                 step_id: str,
                                 dataset: str):
    # Import dataset as pd dataframe excluding the first 150 rows
    json_dataframe = ''
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    df = pd.read_csv(path_to_storage + '/' + dataset, skiprows=150)
    df.drop(df.columns[[12]], axis=1, inplace=True)
    # df = df.set_index('Line')
    df_updated = df.head(100)
    json_dataframe = df_updated.to_json(orient="records")
    return {"dataframe": json_dataframe}
    # df.to_excel(get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + 'initial_dataset.xlsx')

@router.get("/return_final_dataset", tags=["actigraphy_analysis"])
async def return_final_dataset(workflow_id: str,
                                 run_id: str,
                                 step_id: str,
                               dataset: str):
    # path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    df = pd.read_excel(get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + 'new_dataset.xlsx')
    # df.reset_index(inplace=True)
    # print(df)
    df_updated = df.head(100)
    json_dataframe = df_updated.to_json(orient="records")
    return {"dataframe": json_dataframe}

@router.get("/change_activity_status", tags=["actigraphy_analysis"])
async def change_activity_status(workflow_id: str,
                                run_id: str,
                                step_id: str,
                                dataset: str,
                                activity_status: str,
                                start_date: str,
                                end_date: str):
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    raw = pyActigraphy.io.read_raw_rpx(path_to_storage + '/' + dataset)

    # Convert the dates from string to datetime to change their format
    # Date and time in format "YYYY/MM/DD hh:mm:ss"
    start_date = start_date.replace("-", "/")
    end_date = end_date.replace("-", "/")
    format_string = "%Y/%m/%d %H:%M:%S.%f"

    # Convert start date string to date using strptime
    start_date_dt = datetime.strptime(start_date, format_string)
    # Convert end date string to date using strptime
    end_date_dt = datetime.strptime(end_date, format_string)

    # Convert them back to string and format them correctly, so they can be compared with the ones from the dataframe
    start_date_final = start_date_dt.strftime("%d/%m/%Y %H:%M:%S")
    end_date_final = end_date_dt.strftime("%d/%m/%Y %H:%M:%S")

    # Make a copy of the dataset which we will use to make our changes
    file_exists = path_to_storage + '/output/' + 'NewAnalysisCopy.csv'
    isExisting = os.path.exists(file_exists)
    # print(isExisting)
    if (isExisting == False):
        source = path_to_storage + '/' + dataset
        target = path_to_storage + '/output/' + 'NewAnalysisCopy.csv'
        shutil.copyfile(source, target)
    else:
        print("The file already exists!")
    # print(isExisting)
    # Import dataset as pd dataframe excluding the first 150 rows
    df = pd.read_csv(path_to_storage + '/output/' + 'NewAnalysisCopy.csv', skiprows=150)

    # Using DataFrame.apply() and lambda function to join the date and time columns to create a Datetime
    df["Datetime"] = df[["Date", "Time"]].apply(lambda x: " ".join(x), axis=1)

    # Change the Interval status by checking multiple conditions
    df['Interval Status'] = np.where((df['Datetime'] >= start_date_final) & (df['Datetime'] <= end_date_final), activity_status, df['Interval Status'])

    print('The activity status ' + activity_status + ' will be assigned from ' + start_date_final + ' to ' + end_date_final)
    print(df.head(10))

    # Reset the columns to be in the exact format they were before
    df.drop(df.columns[[12, 13]], axis=1, inplace=True)
    df = df.set_index('Line')
    df.to_excel(get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + 'new_dataset.xlsx')
    change_final_csv(workflow_id, run_id, step_id)

# @router.get("/change_final_csv", tags=["actigraphy_analysis_assessment_algorithm"])
def change_final_csv(workflow_id: str,
                                run_id: str,
                                step_id: str):
    subject_properties = pd.DataFrame(columns=['Field', 'Value'])
    found = False
    count = 1
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    df = pd.read_excel(get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + 'new_dataset.xlsx')
    df = df.astype(str)
    # df.set_index('Line', inplace=True)
    print('"' + '","'.join(df.loc[0, :].values.flatten().tolist()) + '"')
    # save the original file in the same path + output and use that one in the line below
    with open(path_to_storage + '/output/' + 'NewAnalysisCopy.csv', 'r') as f:
        get_all = f.readlines()

    with open(path_to_storage + '/output/' + 'NewAnalysisCopy.csv', 'w') as f:
        list_count = 0
        for i, line in enumerate(get_all, 1):
            if i >= 153:
                f.writelines('"' + '","'.join(df.loc[list_count, :].values.flatten().tolist()) + '",' + "\n")
                list_count = list_count + 1
                # print(list_count)
            else:
                f.writelines(line)

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
            '/neurodesktop-storage/runtime_config/workflow_3fa85f64-5717-4562-b3fc-2c963f66afa6/run_3fa85f64-5717-4562-b3fc-2c963f66afa6/step_3fa85f64-5717-4562-b3fc-2c963f66afa6/0345-024_18_07_2022_13_00_00_New_Analysis.csv',
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


@router.get("/return_functional_linear_modelling", tags=["actigraphy_analysis"])
async def return_functional_linear_modelling(workflow_id: str,
                                             run_id: str,
                                             step_id: str,
                                             dataset: str):
    # Define path
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    # Fourier basis expansion (Single)
    try:
        raw = pyActigraphy.io.read_raw_rpx(
            path_to_storage + '/' + dataset,
            start_time='2022-07-18 12:00:00',
            period='7 days'
        )
    except Exception as e:
        print(e)
        return JSONResponse(status_code=500)
    # create objects for layout and traces
    # layout = go.Layout(autosize=False, width=850, height=600, title="", xaxis=dict(title=""), shapes=[],
    #                    showlegend=True)

    # Resampling frequency for the daily activity profile
    freq = '1min'
    # The number of basis functions is max_order*2+1 (1 constant + n cosine functions + n sine functions)
    max_order = 9

    flm = FLM(basis='fourier', sampling_freq=freq, max_order=max_order)
    # By setting the "verbose" parameter to True, the result of least-square fit is displayed:
    flm.fit(raw, verbose=True)
    flm_est = flm.evaluate(raw)
    daily_avg = raw.average_daily_activity(binarize=False, freq=freq)
    # set x-axis labels and their corresponding data values
    labels = ['00:00', '06:00', '12:00', '18:00']
    tickvals = ['00:00:00', '06:00:00', '12:00:00', '18:00:00']

    layout = go.Layout(
        autosize=False, width=1000, height=600,
        title="FML Daily Profile",
        xaxis=dict(
            title="Time of day (HH:MM)",
            ticktext=labels,
            tickvals=tickvals),
        yaxis=dict(title="Counts (a.u)"),
        shapes=[], showlegend=True)

    fig = go.Figure(data=[
            go.Scatter(x=daily_avg.index.astype(str),y=daily_avg,name='Raw activity'),
            go.Scatter(x=daily_avg.index.astype(str),y=flm_est,name='Fourier expansion (9th order)')
        ],layout=layout)
    # fig.show()
    graphJSON = plotly.io.to_json(fig, pretty=True)
    return {"flm_figure": graphJSON}


@router.get("/return_multi_functional_linear_modelling", tags=["actigraphy_analysis"])
async def return_multi_functional_linear_modelling(workflow_id: str,
                                             run_id: str,
                                             step_id: str,
                                             multiple_datasets: str):
    # Define path
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    raw = pyActigraphy.io.read_raw_rpx(
        path_to_storage + '/' + multiple_datasets,
        start_time='2022-07-18 12:00:00',
        period='7 days'
    )
    # Fourier basis expansion (Multi)
    reader = pyActigraphy.io.read_raw(path_to_storage + multiple_datasets + '_example_*.csv', 'RPX', n_jobs=10, prefer='threads', verbose=10)
    # Define a FLM Object that can be (re-)used to fit the data
    flm_fourier = FLM(basis='fourier', sampling_freq='10min', max_order=10)
    # Fit all the recordings contained in the "reader":
    flm_fourier.fit_reader(reader, verbose_fit=True, n_jobs=2, prefer='threads', verbose_parallel=10)

    y_est_group_fourier = flm_fourier.evaluate_reader(reader, r=10, n_jobs=2, prefer='threads', verbose_parallel=10)
    print(y_est_group_fourier.items())
    daily_avg = raw.average_daily_activity(binarize=False, freq='10min')
    # create objects for layout and traces
    # multi_layout = go.Layout(autosize=False, width=850, height=600, title="", xaxis=dict(title=""), shapes=[],
    #                    showlegend=True)
    # set x-axis labels and their corresponding data values
    labels = ['00:00', '06:00', '12:00', '18:00']
    tickvals = ['00:00:00', '06:00:00', '12:00:00', '18:00:00']

    layout = go.Layout(
        autosize=False, width=900, height=600,
        title="Daily profile",
        xaxis=dict(
            title="Time of day (HH:MM)",
            ticktext=labels,
            tickvals=tickvals),
        yaxis=dict(title="Counts (a.u)"),
        shapes=[], showlegend=True)
    multi_fig = go.Figure(data=[go.Scatter(x=daily_avg.index.astype(str),y=v,name=k) for k,v in y_est_group_fourier.items()],layout=layout)
    # multi_fig.show()
    graphJSON = plotly.io.to_json(multi_fig, pretty=True)
    return {"multi_flm_figure": graphJSON}

@router.get("/return_singular_spectrum_analysis", tags=["actigraphy_analysis"])
async def return_singular_spectrum_analysis(workflow_id: str,
                                 run_id: str,
                                 step_id: str,
                                 dataset: str):
    # Define path
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    # Fourier basis expansion (Single)
    try:
        raw = pyActigraphy.io.read_raw_rpx(
            path_to_storage + '/' + dataset,
            start_time='2022-07-18 12:00:00',
            period='7 days'
        )
    except Exception as e:
        print(e)
        return JSONResponse(status_code=500)

    # Scree diagram
    mySSA = SSA(raw.data, window_length='24h')
    # Access the trajectory matrix
    mySSA.trajectory_matrix().shape
    mySSA.fit()
    # By definition, the sum of the partial variances should be equal to 1:
    mySSA.lambda_s.sum()
    layout_1 = go.Layout(
        height=600,
        width=800,
        title="Scree diagram",
        xaxis=dict(title="Singular value index", type='log', showgrid=True, gridwidth=1, gridcolor='LightPink',
                   title_font={"size": 20}),
        yaxis=dict(title=r'$\lambda_{k} / \lambda_{tot}$', type='log', showgrid=True, gridwidth=1,
                   gridcolor='LightPink', ),
        showlegend=False
    )
    fig_1 = go.Figure(data=[go.Scatter(x=np.arange(0, len(mySSA.lambda_s) + 1), y=mySSA.lambda_s)], layout=layout_1)
    # fig_1.show()
    graphJSON1 = plotly.io.to_json(fig_1, pretty=True)

    # Correlation matrix
    w_corr_mat = mySSA.w_correlation_matrix(10)
    fig_2 = go.Figure(data=[go.Heatmap(z=w_corr_mat)], layout=go.Layout(height=800, width=800))
    fig_2.update_layout(title_text="W-correlation matrix for the reconstructed series (10 first elementary matrices)")
    # fig_2.show()
    graphJSON2 = plotly.io.to_json(fig_2, pretty=True)

    # Diagonal averaging
    trend = mySSA.X_tilde(0)
    # By definition, the reconstructed components must have the same dimension as the original signal:
    trend.shape[0] == len(raw.data.index)
    et12 = mySSA.X_tilde([1, 2])
    et34 = mySSA.X_tilde([3, 4])
    layout_3 = go.Layout(
        height=600,
        width=800,
        title="Diagonal averaging",
        xaxis=dict(title='Date Time'),
        yaxis=dict(title='Count'),
        shapes=[],
        showlegend=True
    )
    fig_3 = go.Figure(data=[
        go.Scatter(x=raw.data.index, y=raw.data, name='Activity'),
        go.Scatter(x=raw.data.index, y=trend, name='Trend'),
        go.Scatter(x=raw.data.index, y=trend + et12, name='Circadian component'),
        go.Scatter(x=raw.data.index, y=trend + et34, name='Ultradian component')
    ], layout=layout_3)
    # fig_3.show()
    graphJSON3 = plotly.io.to_json(fig_3, pretty=True)

    # Compare original signal with reconstruction
    rec = mySSA.reconstructed_signal([0, 1, 2, 3, 4, 5, 6])
    fig_4 = go.Figure(data=[
        go.Scatter(x=raw.data.index, y=raw.data, name='Activity'),
        go.Scatter(x=raw.data.index, y=rec, name='Reconstructed signal')
    ], layout=go.Layout(height=600, width=800, title="Compare original signal with its reconstruction", showlegend=True))
    # fig_4.show()
    graphJSON4 = plotly.io.to_json(fig_4, pretty=True)
    return {"ssa_figure_1": graphJSON1, "ssa_figure_2": graphJSON2, "ssa_figure_3": graphJSON3, "ssa_figure_4": graphJSON4}

@router.get("/return_detrended_fluctuation_analysis", tags=["actigraphy_analysis"])
async def return_detrended_fluctuation_analysis(workflow_id: str,
                                 run_id: str,
                                 step_id: str,
                                 dataset: str):
    # Define path
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    # Fourier basis expansion (Single)
    try:
        raw = pyActigraphy.io.read_raw_rpx(
            path_to_storage + '/' + dataset,
            start_time='2022-07-18 12:00:00',
            period='7 days'
        )
    except Exception as e:
        print(e)
        return JSONResponse(status_code=500)

    # Signal detrending and integration (1st step)
    profile = Fractal.profile(raw.data.values)
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter(x=raw.data.index.astype(str), y=raw.data.values, name='Data'),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=raw.data.index.astype(str), y=profile, name='Profile'),
        secondary_y=True,
    )
    # Add figure title
    fig.update_layout(
        title_text="Detrended and integrated data profile", height=800, width=1100
    )
    # Set x-axis title
    fig.update_xaxes(title_text="Date time")
    # Set y-axes titles
    fig.update_yaxes(title_text="Activity counts", secondary_y=False)
    # fig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)
    # fig.show()
    graphJSON1 = plotly.io.to_json(fig, pretty=True)
    # return {"flm_figure": graphJSON}

    # Signal segmentation into non-overlappiong windows (2nd step)
    # Example of segmentation with a window size of 1000 elements.
    n = 1000
    segments = Fractal.segmentation(profile, n)
    len(profile)
    segments.shape

    # Local and global q-th order fluctuations
    local_fluctuations = [Fractal.local_msq_residuals(segment, deg=1) for segment in segments]
    Fractal.q_th_order_mean_square(local_fluctuations, q=2)

    # Calculate and plot DFA
    n_array = np.geomspace(10, 1440, num=50, endpoint=True,
                           dtype=int)  # Numbers spaced evenly on a log scale, ranging from an ultradian time scale (10 min.) to a circadian one (1440 min, i.e. 24h)
    F_n = Fractal.dfa(raw.data, n_array, deg=1)
    fig_2 = go.Figure(data=[
        go.Scatter(x=n_array, y=np.log(F_n), name='Data fluctuation', mode='markers+lines')])
    fig_2.update_layout(
        title_text="Signal segmentation into non-overlapping windows",
        height=800, width=800,
        xaxis=dict(title='Time (min.)', type='log'),
        yaxis=dict(title='log(F(n))')
    )
    # fig_2.show()
    graphJSON2 = plotly.io.to_json(fig_2, pretty=True)
    # Export generalized Hurst component
    Fractal.generalized_hurst_exponent(F_n, n_array, log=False)

    # Repeat the DFA for multiple q orders (not only for q=2)
    # Multi DFA
    q_array = [1, 2, 3, 4, 5, 6]
    MF_F_n = Fractal.mfdfa(raw.data, n_array, q_array, deg=1)
    fig_3 = go.Figure(data=[
        go.Scatter(x=n_array, y=np.log(MF_F_n[:, q]), name='Data fluctuation (q-th order: {})'.format(q_array[q]),
                   mode='markers+lines') for q in range(len(q_array))])
    fig_3.update_layout(
        title_text="Multifractal DFA (various q-orders)",
        height=800, width=800,
        xaxis=dict(title='Time (min.)', type='log'),
        yaxis=dict(title='log(F(n))')
    )
    # fig_3.show()
    graphJSON3 = plotly.io.to_json(fig_3, pretty=True)
    mf_h_q = [Fractal.generalized_hurst_exponent(MF_F_n[:, q], n_array) for q in range(len(q_array))]
    print(mf_h_q)
    return {"dfa_figure_1": graphJSON1, "dfa_figure_2": graphJSON2, "dfa_figure_3": graphJSON3}

@router.get("/return_inactivity_mask_visualisation", tags=["actigraphy_analysis"])
async def return_inactivity_mask_visualisation(workflow_id: str,
                                                step_id: str,
                                                run_id: str,
                                                inactivity_masking_period_hour: str,
                                                inactivity_masking_period_minutes: str,
                                               dataset: str):
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    raw = pyActigraphy.io.read_raw_rpx(
        path_to_storage + '/' + dataset
    )
    print(inactivity_masking_period_hour, inactivity_masking_period_minutes)
    my_duration = inactivity_masking_period_hour + 'h' + inactivity_masking_period_minutes + 'min'
    print(my_duration)
    # Create inactivity mask
    raw.create_inactivity_mask(duration=my_duration)
    layout_vis = go.Layout(title="Actigraphy data", xaxis=dict(title="Date time"), yaxis=dict(title="Counts/period"),
                       showlegend=False)
    layout_mask = go.Layout(title="Data mask", xaxis=dict(title="Date time"), yaxis=dict(title="Mask"), showlegend=False)

    fig = make_subplots(rows=2, cols=1,
                        vertical_spacing = 0.15,
                        subplot_titles = ("Actigraphy Activity", "Inactivity Masking")
                        )
    fig.add_scatter(x=raw.data.index.astype(str), y=raw.data, mode='lines', row=1, col=1)
    fig.add_scatter(x=raw.mask.index.astype(str), y=raw.mask, mode='lines', row=2, col=1)
    fig.update_layout(
        height=800, width=950
    )
    # fig.show()
    graphJSON = plotly.io.to_json(fig, pretty=True)
    return {"visualisation_inactivity_mask": graphJSON}

@router.get("/return_add_mask_period", tags=["actigraphy_analysis"])
async def return_add_mask_period(workflow_id: str,
                                step_id: str,
                                run_id: str,
                                 mask_period_start: str,
                                 mask_period_end: str,
                                 dataset: str):
    original = r'example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv'
    target = r'example_data/actigraph/dataset_copy.csv'
    shutil.copyfile(original, target)

    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    file_exists = path_to_storage + '/output/' + 'AddMaskPeriodCopy.csv'
    isExisting = os.path.exists(file_exists)
    # print(isExisting)
    if (isExisting == False):
        source = path_to_storage + '/' + dataset
        target = path_to_storage + '/output/' + 'AddMaskPeriodCopy.csv'
        shutil.copyfile(source, target)
    else:
        print("The file already exists!")
    raw = pyActigraphy.io.read_raw_rpx(
        path_to_storage + '/output/' + 'AddMaskPeriodCopy.csv'
    )
    # Create inactivity mask
    raw.add_mask_period(start=mask_period_start, stop=mask_period_end)
    fig = make_subplots(rows=2, cols=1,
                        vertical_spacing = 0.15,
                        subplot_titles = ("Actigraphy Activity", "Inactivity Masking")
                        )
    fig.add_scatter(x=raw.data.index.astype(str), y=raw.data, mode='lines', row=1, col=1)
    fig.add_scatter(x=raw.mask.index.astype(str), y=raw.mask, mode='lines', row=2, col=1)
    fig.update_layout(
        height=800, width=950
    )
    # fig.show()
    print(raw.IS() )
    raw.mask_inactivity = True
    print (raw.IS() )
    # raw.mask_inactivity = False
    graphJSON = plotly.io.to_json(fig, pretty=True)
    return {"visualisation_add_mask_period": graphJSON}

  
@router.get("/actigraphy_metrics", tags=["actigraphy_analysis"])
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
        # raw = pyActigraphy.io.read_raw_rpx('/neurodesktop-storage/runtime_config/workflow_3fa85f64-5717-4562-b3fc-2c963f66afa6/run_3fa85f64-5717-4562-b3fc-2c963f66afa6/step_3fa85f64-5717-4562-b3fc-2c963f66afa6/0345-024_18_07_2022_13_00_00_New_Analysis.csv')
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

        tbl_res=[]

        pRA, pRA_weights = raw.pRA(threshold=threshold, start=None, period=period)
        pAR, pAR_weights = raw.pAR(threshold=threshold, start=None, period=period)

        df = pd.DataFrame()
        df['pRA'] = raw.pRA(threshold=threshold, start=None, period=period)[0]
        df['pRA_weights'] = raw.pRA(threshold=threshold, start=None, period=period)[1]
        df['t']=df.index
        df1 = pd.DataFrame()
        df1['pAR'] = raw.pAR(threshold=threshold, start=None, period=period)[0]
        df1['pAR_weights'] = raw.pAR(threshold=threshold, start=None, period=period)[1]
        df1['t'] = df1.index

        temp_to_append = {'id': 1,
                          "Name": raw.name,
                          "Start_time": (raw.start_time).__str__(),
                          "Duration": (raw.duration()).__str__(),
                          "Serial": raw.uuid,
                          "frequency": (raw.frequency).__str__(),
                          "IS": raw.IS(binarize=binarize, threshold=threshold, freq=freq),
                          "ISm": raw.ISm(binarize=binarize, threshold=threshold),
                          'ISp':raw.ISp(binarize=binarize, threshold=threshold, period=period),
                          "IV": raw.IV(binarize=binarize, threshold=threshold, freq=freq),
                          "IVm": raw.IVm(binarize=binarize, threshold=threshold),
                          'IVp': raw.IVp(binarize=binarize, threshold=threshold, period=period),
                          "L5": raw.L5(binarize=binarize, threshold=threshold),
                          "L5p": raw.L5p(binarize=binarize, threshold=threshold, period=period),
                          "M10": raw.M10(binarize=binarize, threshold=threshold),
                          "M10p": raw.M10p(binarize=binarize, threshold=threshold,period=period),
                          "RA": raw.RA(binarize=binarize, threshold=threshold),
                          "RAp": raw.RAp(binarize=binarize, threshold=threshold,period=period),
                          'pRA': df.to_json(orient='records'),
                          'pAR': df1.to_json(orient='records'),
                          "ADAT": raw.ADAT(binarize=binarize, threshold=threshold),
                          "ADATp": raw.ADATp(binarize=binarize, threshold=threshold,period=period),
                          "kRA": raw.kRA(threshold=threshold, start=None, period=period),
                          "kAR": raw.kAR(threshold=threshold, start=None, period=period),
                          }
        tbl_res.append(temp_to_append)
        layout = go.Layout(title="",xaxis=dict(title=""), showlegend=False)
        layout.update(title="Rest->Activity transition probability", xaxis=dict(title="Time [min]"), showlegend=False);
        output = go.Figure(data=go.Scatter(x=pRA.index, y=pRA, name='', mode='markers'), layout=layout)
        pio.write_image(output, get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + 'pRA.svg')

        layout.update(title="Activity->Rest transition probability", xaxis=dict(title="Time [min]"), showlegend=False);
        output = go.Figure(data=go.Scatter(x=pAR.index, y=pAR, name='', mode='markers'), layout=layout)
        pio.write_image(output, get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + 'pAR.svg')

        test_status = 'Unable to create info file.'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data['results'] |= {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Actigraphy Metrics',
                "test_params": {
                    'file': file,
                    'freq_offset': freq_offset,
                    'number_of_offsets': number_of_offsets,
                    'period_offset': period_offset,
                    'number_of_periods':number_of_periods,
                    'binarize': binarize,
                    'threshold': threshold
                },
                "test_results": {
                    'metrics':tbl_res
                },
                "Output_datasets":[],
                'Saved_plots': []
            }
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'Result': tbl_res},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'Result': {}},
                            status_code=200)

@router.get("/cosinor_analysis_initial_values", tags=["actigraphy_analysis"])
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


@router.get("/cosinor_analysis", tags=["actigraphy_analysis"])
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
        # raw = pyActigraphy.io.read_raw_rpx('/neurodesktop-storage/runtime_config/workflow_3fa85f64-5717-4562-b3fc-2c963f66afa6/run_3fa85f64-5717-4562-b3fc-2c963f66afa6/step_3fa85f64-5717-4562-b3fc-2c963f66afa6/0345-024_18_07_2022_13_00_00_New_Analysis.csv')
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
        test_status = 'Unable to get Cosinor values.'
        print(raw.data.index)
        print(data)
        results = cosinor_obj.fit(raw.data, verbose=True)
        print(results)
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
                "test_name": 'Cosinor fit',
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
