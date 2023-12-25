#!/usr/bin/env python3
import PySimpleGUI as sg
from PIL import Image, ImageTk
import cv2
import numpy as np
import os.path
import time

# Initialize a variable to store image data
image_data = None

def author_window():
    layout = [
        [sg.Text("Foucault KnifeEdge Shadowgram Analyzer (FKESA) Version 2", size=(60, 1), justification="center", font=('Times New Roman', 10, 'bold'), key="-APP-")],
        [sg.HorizontalSeparator()],  # Separator 
        [sg.Text("Author: ", size=(8, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-AUTHOR-"),sg.Text('Pratik M. Tambe <enthusiasticgeek@gmail.com>')],
        [sg.Text("FKESA: ", size=(8, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-VERSION-"),sg.Text(' Version 2.1')],
        [sg.Text("Release Date: ", size=(14, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-RELEASE DATE-"),sg.Text('December 25, 2023')],
        [sg.Text("Credits: ", size=(8, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-AUTHOR-")],
        [sg.Text('Guy Brandenburg, Alan Tarica - National Capital Astronomers (NCA)')],
        [sg.Text('Amateur Telescope Making (ATM) workshop')],
        [sg.Text('Washington D.C., U.S.A.')],
        [sg.Text('Inputs: Alin Tolea, PhD - System Engineer, Human Spaceflight Communication and Tracking Network, NASA Goddard Space Flight Center')],
        [sg.Button('Close')]
    ]

    window = sg.Window('About', layout)

    while True:
        event, _ = window.read()
        if event == sg.WINDOW_CLOSED or event == 'Close':
            window.close()
            break


# Ref: https://stackoverflow.com/questions/57577445/list-available-cameras-opencv-python
def list_available_cameras():
    print("=============================================")
    print("PLEASE WAIT....DETECTING AVAILABLE CAMERAS...")
    print("=============================================")
    CAM_DEBUG=False
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
            if CAM_DEBUG==True:
               print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                if CAM_DEBUG==True:
                   print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                if CAM_DEBUG==True:
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

    # Define the menu items
    menu_def = [
            ['&File', ['---', '&Exit']],
            ['&Help', ['&About']]
    ]

    # First the window layout in 2 columns
    file_list_column = [
        [
            sg.Text("Image Folder",font=('Times New Roman', 12, 'bold'), text_color="navyblue"), 
            sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
            sg.FolderBrowse(),
        ],
        [
            sg.Listbox(
                values=[], enable_events=True, size=(40, 30), key="-FILE LIST-"
            )
        ],
    ]

    # For now will only show the name of the file that was chosen
    image_viewer_column = [
        [sg.Text("Choose an image from list on left:", font=('Times New Roman', 14, 'bold'), text_color="navyblue")],
        [sg.Text(size=(40, 1), key="-TOUT-")],
        [sg.Image(key="-LOAD IMAGE-",size=(320,240))],
    ]


    # Define the window layout
    layout = [
        [sg.Text("FOUCAULT KNIFE-EDGE SHADOWGRAM ANALYZER (FKESA) GUI VERSION 2", size=(100, 1), justification="center", font=('Times New Roman', 14, 'bold'),text_color='darkgreen')],
        [sg.Menu(menu_def, background_color='lightblue',text_color='navy', disabled_text_color='yellow', font='Verdana', pad=(10,10))],
        [sg.HorizontalSeparator()],  # Separator 
        [sg.Image(filename="", key="-IMAGE-", size=(640,480)), sg.VerticalSeparator(), sg.Column(file_list_column), sg.VerticalSeparator(), sg.Column(image_viewer_column),],
        [
            [sg.Text("SELECT CAMERA", size=(50, 1), justification="left", font=('Times New Roman', 12, 'bold'), text_color='navyblue')],
            [sg.HorizontalSeparator()],  # Separator 
            [sg.DropDown(working_ports, default_value='0', key='-CAMERA SELECT-')],
            [sg.Button('OK'), sg.VerticalSeparator(), sg.Button('Cancel')]
        ],
        [sg.HorizontalSeparator()],  # Separator 
        [sg.Text("CIRCULAR HOUGH TRANSFORM PARAMETERS [MIRROR DETECTION]", size=(60, 1), justification="left", font=('Times New Roman', 12, 'bold'), text_color='navyblue')],
        [sg.HorizontalSeparator()],  # Separator 
        [
            sg.Text("MIN DIST (PIXELS), DELAY MILSEC [DEFAULT: 50/1000]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-MINDIST-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (0, 255),
                50,
                1,
                orientation="h",
                size=(50, 15),
                key="-MINDIST SLIDER-",
                font=('Times New Roman', 10, 'bold'),
            ),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (500, 10000),
                500,
                500,
                orientation="h",
                size=(50, 15),
                key="-FRAMES SLIDER-",
                font=('Times New Roman', 10, 'bold'),
            ),

        ],
        [
            sg.Text("PARAMETERS 1 AND 2 (PIXELS) [DEFAULT: 25/60]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-PARAMS-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (0, 255),
                25,
                1,
                orientation="h",
                size=(50, 15),
                key="-PARAM SLIDER A-",
                font=('Times New Roman', 10, 'bold'),
            ),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (0, 255),
                60,
                1,
                orientation="h",
                size=(50, 15),
                key="-PARAM SLIDER B-",
                font=('Times New Roman', 10, 'bold'),
            ),
        ],
        [
            sg.Text("MIN AND MAX RADIUS (PIXELS) [DEFAULT: 10/0]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-RADIUS-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (0, 255),
                20,
                1,
                orientation="h",
                size=(50, 15),
                key="-RADIUS SLIDER A-",
                font=('Times New Roman', 10, 'bold'),
            ),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (0, 255),
                60,
                1,
                orientation="h",
                size=(50, 15),
                key="-RADIUS SLIDER B-",
                font=('Times New Roman', 10, 'bold'),
            ),
        ],
        [sg.HorizontalSeparator()],  # Separator 
        [sg.Text("INTENSITY PARAMETERS [NULL ZONES IDENTIFICATION]", size=(50, 1), justification="left", font=('Times New Roman', 12, 'bold'), text_color='navyblue')],
        [sg.HorizontalSeparator()],  # Separator 
        [
            sg.Text("BRIGHTNESS TOLERANCE AND ZONES [DEFAULT: 10/50]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-INTENSITY PARAMS-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (0, 50),
                10,
                1,
                orientation="h",
                size=(50, 15),
                key="-BRIGHTNESS SLIDER-",
                font=('Times New Roman', 10, 'bold'),
            ),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (30, 50),
                60,
                1,
                orientation="h",
                size=(50, 15),
                key="-ZONES SLIDER-",
                font=('Times New Roman', 10, 'bold'),
            ),
        ],
        [
            sg.Text("ANGLE OF BRIGHTNESS SLICE (DEGREES) [DEFAULT: 10]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-ANGLE-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (10, 90),
                10,
                1,
                orientation="h",
                size=(50, 15),
                key="-ANGLE SLIDER-",
                font=('Times New Roman', 10, 'bold'),
            ),
            sg.VerticalSeparator(),  # Separator 
        ],
        [sg.HorizontalSeparator()],  # Separator 
        [sg.Text("PRIMARY MIRROR PARAMETERS [PARABOLIC MIRROR or K = -1]", size=(60, 1), justification="left", font=('Times New Roman', 12, 'bold'), text_color='navyblue')],
        [sg.HorizontalSeparator()],  # Separator 
        [
            sg.Text("DIAMETER AND FOCAL LENGTH (inches) [DEFAULT: 6/48]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-MIRROR PARAMS-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (1, 255),
                6,
                0.25,
                orientation="h",
                size=(50, 15),
                key="-DIAMETER SLIDER-",
                font=('Times New Roman', 10, 'bold'),
            ),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (1, 255),
                48,
                0.25,
                orientation="h",
                size=(50, 15),
                key="-FOCAL LENGTH SLIDER-",
                font=('Times New Roman', 10, 'bold'),
            ),
        ],
        [sg.Button("Exit", size=(10, 1)), sg.VerticalSeparator(), sg.Button("About", size=(10, 1)), sg.VerticalSeparator(), sg.Button("Save Image", size=(15, 1)) ],
    ]

    #sg.theme_previewer()
    try:
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
                elif event == 'Save Image':
                    if image_data is not None:
                        # Use OpenCV to write the image data to a file
                        filename = f"fkesa_v2_{int(time.time())}.png"  # Generate a filename (you can adjust this)
                        with open(filename, 'wb') as f:
                            f.write(image_data)
                        sg.popup(f"Image saved as: {filename}")
                # Folder name was filled in, make a list of files in the folder
                elif event == "-FOLDER-":
                    folder = values["-FOLDER-"]
                    try:
                        # Get list of files in folder
                        file_list = os.listdir(folder)
                    except:
                        file_list = []

                    fnames = [
                        f
                        for f in file_list
                        if os.path.isfile(os.path.join(folder, f))
                        and f.lower().endswith((".bmp",".jpg",".svg",".jpeg",".png", ".gif"))
                    ]
                    window["-FILE LIST-"].update(fnames)
                # Inside your event loop where you update the image element
                elif event == "-FILE LIST-":  # A file was chosen from the listbox
                    try:
                      filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
                      window["-TOUT-"].update(filename)
                      image = Image.open(filename)
                      image.thumbnail((400, 400))  # Resize image if needed
                      bio = ImageTk.PhotoImage(image)
                      window["-LOAD IMAGE-"].update(data=bio)
                    except Exception as e:
                      print(e)
                    except:
                        pass

                elif event == 'About':
                    #window.hide()  # Hide the main window
                    author_window()  # Open the author information window
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
                image_data = imgbytes  # Update the image data variable

                # Sleep for 0.5 milliseconds (500 microseconds)
                milliseconds = 100 / 1000
                time.sleep(milliseconds)

            cap.release()
            window.close()

    except Exception as e:
            print(f"An error occurred: {e}")

main()
