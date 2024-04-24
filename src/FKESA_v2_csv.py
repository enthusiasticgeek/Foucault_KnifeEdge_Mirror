#!/usr/bin/env python3
#Author: Pratik M Tambe <enthusiasticgeek@gmail.com>
#Date: Jan 10, 2024

import PySimpleGUI as sg
import csv

#TODO(Pratik) Add exceptions
# Function to read CSV file
def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            data.append(row)
    return data

def read_csv_v1(filename):
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        #Skip first 2 rows that are header
        next(csv_reader)  # Skip the first row
        next(csv_reader)  # Skip the second row
        for row in csv_reader:
            data.append(row)  # Read all columns
    return data

# Function to truncate text in columns
def truncate_columns(data, max_width):
    truncated_data = []
    for row in data:
        truncated_row = [cell[:max_width] for cell in row]
        truncated_data.append(truncated_row)
    return truncated_data

sg.theme("LightBlue")
# Define layout
layout = [
    [sg.Text('Select a FKESA CSV file')],
    [sg.Input(key='-FILE-'), sg.FileBrowse(file_types=(("CSV Files", "*.csv"),))],
    [sg.Button('Open'), sg.Button('Exit')]
]

# Create the window
window = sg.Window('FKESA v2 Measurement Data Viewer', layout)

while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == 'Exit':
        break
    elif event == 'Open':
        filename = values['-FILE-']
        if filename:
            #data = read_csv(filename)
            data = read_csv_v1(filename)
            headers = data[0]  # Assuming the first row contains headers
            table_data = data[1:]  # Data excluding headers
            # Truncate each cell to a maximum width of 30 characters
            truncated_data = truncate_columns(table_data, 30)
            # Create layout for the data window
            """
            data_layout = [
                [sg.Table(values=truncated_data, headings=headers, display_row_numbers=True,
                          auto_size_columns=True, justification='center', enable_events=True,
                          num_rows=min(len(truncated_data), 30), alternating_row_color='navyblue')]
            ]
            """
            # Create layout for the data window
            data_layout = [
		    [sg.Table(
			values=truncated_data,
			headings=headers,
			display_row_numbers=True,
			auto_size_columns=True,
			justification='center',
			num_rows=min(len(truncated_data), 50),
			alternating_row_color='yellow',
			key='-TABLE-',
			vertical_scroll_only=False
		    )],
            ]

            # Create the data window
            data_window = sg.Window('FKESA v2 Measurement Data', data_layout)

            while True:
                data_event, _ = data_window.read()
                if data_event == sg.WINDOW_CLOSED:
                    break
            data_window.close()
window.close()

