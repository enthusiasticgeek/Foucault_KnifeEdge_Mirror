#!/usr/bin/env python3
import PySimpleGUI as sg
import cv2
import numpy as np
import time

# Ref: https://stackoverflow.com/questions/57577445/list-available-cameras-opencv-python
def list_available_cameras():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 2: # if there are more than 2 non working ports stop the testing. 
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports,non_working_ports

def main():
    sg.theme("LightBlue")
    list_available_cameras()
    # Fetch available cameras
    available_ports, working_ports, non_working_ports = list_available_cameras()
    print(available_ports)

    # Define the window layout
    layout = [
        [sg.Text("FOUCAULT KNIFE-EDGE SHADOWGRAM ANALYZER (FKESA)", size=(100, 1), justification="center", font=('Times New Roman', 14, 'normal'),text_color='darkgreen')],
        [sg.HorizontalSeparator()],  # Separator 
        [sg.Image(filename="", key="-IMAGE-")],
        [
            [sg.Text('Select Camera:')],
            [sg.DropDown(working_ports, key='-CAMERA SELECT-')],
            [sg.Button('OK'), sg.Button('Cancel')]
        ],
        [sg.HorizontalSeparator()],  # Separator 
        [sg.Text("Circular Hough Transform Parameters", size=(50, 1), justification="left", font=('Times New Roman', 12, 'bold'), text_color='darkblue')],
        [
            sg.Text("Min Dist (pixels), Delay msec [Default: 50/1000]", size=(50, 1), justification="left", font=('Times New Roman', 12, 'normal'), key="-MINDIST-"),
            sg.Slider(
                (0, 255),
                50,
                1,
                orientation="h",
                size=(50, 15),
                key="-MINDIST SLIDER-",
                font=('Times New Roman', 10, 'normal'),
            ),
            sg.Slider(
                (500, 10000),
                500,
                500,
                orientation="h",
                size=(50, 15),
                key="-FRAMES SLIDER-",
                font=('Times New Roman', 10, 'normal'),
            ),

        ],
        [
            sg.Text("Parameters 1 and 2 (pixels) [Default: 25/60]", size=(50, 1), justification="left", font=('Times New Roman', 12, 'normal'), key="-PARAMS-"),
            sg.Slider(
                (0, 255),
                25,
                1,
                orientation="h",
                size=(50, 15),
                key="-PARAM SLIDER A-",
            ),
            sg.Slider(
                (0, 255),
                60,
                1,
                orientation="h",
                size=(50, 15),
                key="-PARAM SLIDER B-",
            ),
        ],
        [
            sg.Text("Min and Max Radius (pixels) [Default: 10/0]", size=(50, 1), justification="left", font=('Times New Roman', 12, 'normal'), key="-RADIUS-"),
            sg.Slider(
                (0, 255),
                20,
                1,
                orientation="h",
                size=(50, 15),
                key="-RADIUS SLIDER A-",
            ),
            sg.Slider(
                (0, 255),
                60,
                1,
                orientation="h",
                size=(50, 15),
                key="-RADIUS SLIDER B-",
            ),
        ],
        [sg.HorizontalSeparator()],  # Separator 
        [sg.Text("Intensity Parameters", size=(50, 1), justification="left", font=('Times New Roman', 12, 'bold'), text_color='darkblue')],
        [
            sg.Text("Brightness Tolerance and Zones [Default: 10/50]", size=(50, 1), justification="left", font=('Times New Roman', 12, 'normal'), key="-INTENSITY PARAMS-"),
            sg.Slider(
                (0, 50),
                10,
                1,
                orientation="h",
                size=(50, 15),
                key="-BRIGHTNESS SLIDER-",
            ),
            sg.Slider(
                (30, 50),
                60,
                1,
                orientation="h",
                size=(50, 15),
                key="-ZONES SLIDER-",
            ),
        ],
        [
            sg.Text("Angle of Brightness Slice (degrees) [Default: 10]", size=(50, 1), justification="left", font=('Times New Roman', 12, 'normal'), key="-ANGLE-"),
            sg.Slider(
                (10, 90),
                10,
                1,
                orientation="h",
                size=(50, 15),
                key="-ANGLE SLIDER-",
                font=('Times New Roman', 10, 'normal'),
            ),
        ],
        [sg.HorizontalSeparator()],  # Separator 
        [sg.Text("Primary Mirror Parameters [Parabolic Mirror]", size=(50, 1), justification="left", font=('Times New Roman', 12, 'bold'), text_color='darkblue')],
        [
            sg.Text("Diameter and Focal Length (inches) [Default: 6/48]", size=(50, 1), justification="left", font=('Times New Roman', 12, 'normal'), key="-MIRROR PARAMS-"),
            sg.Slider(
                (1, 255),
                6,
                0.25,
                orientation="h",
                size=(50, 15),
                key="-DIAMETER SLIDER-",
            ),
            sg.Slider(
                (1, 255),
                48,
                0.25,
                orientation="h",
                size=(50, 15),
                key="-FOCAL LENGTH SLIDER-",
            ),
        ],
        [sg.Button("Exit", size=(10, 1))],
    ]

    # Create the window and show it without the plot
    window = sg.Window("FKESA v2 GUI [LIVE]", layout, location=(800, 400))
    selected_camera = 0
    cap = cv2.VideoCapture(selected_camera)

    # Setting the desired resolution (640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


    while True:
        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == 'OK':
            selected_camera = values['-CAMERA SELECT-']
            print(f"Camera selected: {selected_camera}")

            cap.release()
            cap = cv2.VideoCapture(selected_camera)

            # Setting the desired resolution (640x480)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        ret, frame = cap.read()
        if frame is None:
           continue
       

        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)
        # Sleep for 0.5 milliseconds (500 microseconds)
        milliseconds = 100 / 1000
        time.sleep(milliseconds)

    cap.release()
    window.close()

main()
