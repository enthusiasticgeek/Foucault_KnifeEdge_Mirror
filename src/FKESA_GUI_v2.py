#!/usr/bin/env python3
import cv2
import PySimpleGUI as sg
from PIL import Image, ImageTk
import numpy as np
import os.path
import time
import threading
from FKESA_v2_core import FKESABuilder  # Replace 'fkesa_builder_module' with your module name

# Initialize a variable to store image data
#image_data = None
process_fkesa = False
selected_camera=0
shared_frame = None
cap = None
# Create synchronization events
exit_event = threading.Event()  # Event to signal thread exit
is_playing = True

mindist_val = 50
param_a_val = 25 
param_b_val = 60
radius_a_val = 10
radius_b_val = 0
brightness_tolerance_val = 10
zones_val = 50
angle_val = 10
diameter_val = 6.0
focal_length_val = 48.0
gradient_intensity_val = 3
skip_zones_val = 10
raw_video = True
color_video = True
fkesa_time_delay = 1
current_time = time.time()

def author_window():
    layout = [
        [sg.Text("Foucault KnifeEdge Shadowgram Analyzer (FKESA) Version 2", size=(60, 1), justification="center", font=('Times New Roman', 10, 'bold'), key="-APP-")],
        [sg.HorizontalSeparator()],  # Separator 
        [sg.Text("Author: ", size=(8, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-AUTHOR-"),sg.Text('Pratik M. Tambe <enthusiasticgeek@gmail.com>')],
        [sg.Text("FKESA: ", size=(8, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-VERSION-"),sg.Text(' Version 2.1')],
        [sg.Text("Release Date: ", size=(14, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-RELEASE DATE-"),sg.Text('December 25, 2023')],
        [sg.Text("Credits/Feedback: ", size=(18, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-AUTHOR-")],
        [sg.Text('Guy Brandenburg, Alan Tarica - National Capital Astronomers (NCA)')],
        [sg.Text('Amateur Telescope Making (ATM) workshop')],
        [sg.Text('Washington D.C., U.S.A.')],
        [sg.Text("Suggestions/Feedback: ", size=(25, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-AUTHOR-")],
        [sg.Text('Alin Tolea, PhD - System Engineer, NASA Goddard Space Flight Center')],
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
        if camera is not None:
           camera.release()
    return available_ports,working_ports,non_working_ports


# Create a lock
lock = threading.Lock()
# Flag to indicate if processing_frames thread is running
processing_frames_running = True

def process_frames():
    global processing_frames_running
    global selected_camera
    global cap
    global exit_event
    global current_time
    global is_playing
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(selected_camera)  # Open the default camera
    # Setting the desired resolution (640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
       
    #while processing_frames_running==True:
    while not exit_event.is_set():
        try:
            if cap is None:
               cv2.destroyAllWindows()
               return

            ret, frame = cap.read()

            # Setting the desired resolution (640x480)
            #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if frame is None:
                continue
            
            # Acquire the lock before updating the shared resource
            with lock:
                # Your image processing logic here
                # For example, convert frame to grayscale
                #fkesa_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                global mindist_val
                global param_a_val 
                global param_b_val
                global radius_a_val
                global radius_b_val
                global brightness_tolerance_val
                global zones_val
                global angle_val
                global diameter_val
                global focal_length_val
                global raw_video
                global color_video
                global gradient_intensity_val
                global skip_zones_val
                global fkesa_time_delay
                preprocess_frame = None
                # color or grayscale?
                if color_video == False:
                        preprocess_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                        preprocess_frame = frame

                # color or grayscale?
                if raw_video == True:
                        fkesa_frame = preprocess_frame
                else:
                        """
                        start_time = time.time()
                        """
                        builder = FKESABuilder()
                        builder.with_folder('output_folder')
                        builder.with_param('minDist', mindist_val)
                        builder.with_param('param1', param_a_val)
                        builder.with_param('param2', param_b_val)
                        builder.with_param('minRadius', radius_a_val)
                        builder.with_param('maxRadius', radius_b_val)
                        builder.with_param('brightnessTolerance', brightness_tolerance_val)
                        builder.with_param('roiAngleDegrees', zones_val)
                        builder.with_param('zones', zones_val)
                        builder.with_param('mirrorDiameterInches', diameter_val)
                        builder.with_param('mirrorFocalLengthInches', focal_length_val)
                        builder.with_param('gradientIntensityChange', gradient_intensity_val)
                        builder.with_param('skipZonesFromCenter', skip_zones_val)
                        # ... Include other parameters as needed

                        # Build and execute the operation
                        fkesa_frame = builder.build(frame)
                        time.sleep(fkesa_time_delay)
                        """
                        end_time = time.time()
                        time_diff_seconds = end_time - start_time
                        time_diff_milliseconds = time_diff_seconds * 1000  # Convert to milliseconds
                        time_diff_microseconds = time_diff_seconds * 1000000  # Convert to microseconds
                        
                        print(f"Time taken (seconds): {time_diff_seconds}")
                        print(f"Time taken (milliseconds): {time_diff_milliseconds}")
                        print(f"Time taken (microseconds): {time_diff_microseconds}")
                        # Sleep for remaining time (5 seconds - time taken for process execution)
                        time.sleep(max(0, fkesa_time_delay - time_diff_seconds))
                        """

                if fkesa_frame is None:
                   continue

                global shared_frame
                if fkesa_frame is not None:
                   shared_frame = fkesa_frame.copy()
        except Exception as e:
               print(f"An exception occurred {e}")
      

    cap.release()
    cv2.destroyAllWindows()

try:

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
        [sg.Image(key="-LOAD IMAGE-",size=(200,200))],
    ]


    # Define the window layout
    layout = [
        [sg.Text("FOUCAULT KNIFE-EDGE SHADOWGRAM ANALYZER (FKESA) GUI VERSION 2", size=(100, 1), justification="center", font=('Times New Roman', 14, 'bold'),text_color='darkgreen')],
        [sg.Menu(menu_def, background_color='lightblue',text_color='navy', disabled_text_color='yellow', font='Verdana', pad=(10,10))],
        [sg.HorizontalSeparator()],  # Separator 
        [sg.Image(filename="", key="-IMAGE-", size=(640,480)), sg.VerticalSeparator(), sg.Column(file_list_column), sg.VerticalSeparator(), sg.Column(image_viewer_column),],
        [
            [sg.Text("SELECT CAMERA", size=(50, 1), justification="left", font=('Times New Roman', 12, 'bold'), text_color='navyblue'), sg.VerticalSeparator(), sg.Button('PAUSE VIDEO', key='-PAUSE PLAY VIDEO-',button_color = ('white','green')) ],
            [sg.HorizontalSeparator()],  # Separator 
            #[sg.DropDown(working_ports, default_value='0', enable_events=True, key='-CAMERA SELECT-')],
            [sg.DropDown(working_ports, default_value='0', enable_events=True, key='-CAMERA SELECT-'), sg.VerticalSeparator(), sg.Checkbox('RAW VIDEO', default=True, enable_events=True, key='-RAW VIDEO SELECT-'), sg.VerticalSeparator(), sg.Checkbox('COLORED RAW VIDEO', default=True, enable_events=True, key='-COLOR VIDEO SELECT-'), 
            sg.VerticalSeparator(),  # Separator 
            ],
            [sg.Button('OK'), sg.VerticalSeparator(), sg.Button('Cancel')]
        ],
        [sg.HorizontalSeparator()],  # Separator 
        [sg.Text("CIRCULAR HOUGH TRANSFORM PARAMETERS [MIRROR DETECTION]", size=(60, 1), justification="left", font=('Times New Roman', 12, 'bold'), text_color='navyblue')],
        [sg.HorizontalSeparator()],  # Separator 
        [
            sg.Text("MIN DIST (PIXELS) [DEFAULT: 50]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-MINDIST A-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (0, 255),
                50,
                1,
                orientation="h",
                enable_events=True,
                size=(50, 15),
                key="-MINDIST SLIDER-",
                font=('Times New Roman', 10, 'bold'),
            ),
            sg.VerticalSeparator(),  # Separator 
            sg.Text("PROCESSING DELAY SECONDS [DEFAULT: 1]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-MINDIST B-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (0,3),
                1,
                1,
                orientation="h",
                enable_events=True,
                size=(50, 15),
                key="-DELAY SLIDER-",
                font=('Times New Roman', 10, 'bold'),
            ),
            sg.VerticalSeparator(),  # Separator 
        ],
        [
            sg.Text("PARAMETERS 1 (PIXELS) [DEFAULT: 25]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-PARAMS A-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (0, 255),
                25,
                1,
                orientation="h",
                enable_events=True,
                size=(50, 15),
                key="-PARAM SLIDER A-",
                font=('Times New Roman', 10, 'bold'),
            ),
            sg.VerticalSeparator(),  # Separator 
            sg.Text("PARAMETER 2 (PIXELS) [DEFAULT: 60]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-PARAMS B-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (0, 255),
                60,
                1,
                orientation="h",
                enable_events=True,
                size=(50, 15),
                key="-PARAM SLIDER B-",
                font=('Times New Roman', 10, 'bold'),
            ),
            sg.VerticalSeparator(),  # Separator 
        ],
        [
            sg.Text("MIN RADIUS (PIXELS) [DEFAULT: 10]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-MIN RADIUS-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (0, 255),
                20,
                1,
                orientation="h",
                enable_events=True,
                size=(50, 15),
                key="-RADIUS SLIDER A-",
                font=('Times New Roman', 10, 'bold'),
            ),
            sg.VerticalSeparator(),  # Separator 
            sg.Text("MAX RADIUS (PIXELS) [DEFAULT: 0]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-MAX RADIUS-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (0, 255),
                0,
                1,
                orientation="h",
                enable_events=True,
                size=(50, 15),
                key="-RADIUS SLIDER B-",
                font=('Times New Roman', 10, 'bold'),
            ),
            sg.VerticalSeparator(),  # Separator 
        ],
        [sg.HorizontalSeparator()],  # Separator 
        [sg.Text("INTENSITY PARAMETERS [NULL ZONES IDENTIFICATION]", size=(50, 1), justification="left", font=('Times New Roman', 12, 'bold'), text_color='navyblue')],
        [sg.HorizontalSeparator()],  # Separator 
        [
            sg.Text("BRIGHTNESS TOLERANCE [DEFAULT: 10]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-INTENSITY PARAMS A-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (0, 50),
                10,
                1,
                orientation="h",
                enable_events=True,
                size=(50, 15),
                key="-BRIGHTNESS SLIDER-",
                font=('Times New Roman', 10, 'bold'),
            ),
            sg.VerticalSeparator(),  # Separator 
            sg.Text("NUMBER OF ZONES [DEFAULT: 50]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-INTENSITY PARAMS B-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (30, 50),
                60,
                1,
                orientation="h",
                enable_events=True,
                size=(50, 15),
                key="-ZONES SLIDER-",
                font=('Times New Roman', 10, 'bold'),
            ),
            sg.VerticalSeparator(),  # Separator 
        ],
        [
            sg.Text("ANGLE (SLICE OR PIE) (DEGREES) [DEFAULT: 10]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-ANGLE-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (10, 90),
                10,
                1,
                orientation="h",
                size=(50, 15),
                enable_events=True,
                key="-ANGLE SLIDER-",
                font=('Times New Roman', 10, 'bold'),
            ),
            sg.VerticalSeparator(),  # Separator 
            sg.Text("SKIP ZONES FROM THE MIRROR CENTER [DEFAULT: 10]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-SKIP ZONES-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (1, 20),
                10,
                1,
                orientation="h",
                size=(50, 15),
                enable_events=True,
                key="-SKIP ZONES SLIDER-",
                font=('Times New Roman', 10, 'bold'),
            ),
            sg.VerticalSeparator(),  # Separator 
        ],
        [sg.HorizontalSeparator()],  # Separator 
        [sg.Text("PRIMARY MIRROR PARAMETERS [PARABOLIC MIRROR or K = -1]", size=(60, 1), justification="left", font=('Times New Roman', 12, 'bold'), text_color='navyblue')],
        [sg.HorizontalSeparator()],  # Separator 
        [
            sg.Text("DIAMETER (inches) [DEFAULT: 6]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-MIRROR PARAMS A-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (1, 255),
                6,
                0.25,
                orientation="h",
                enable_events=True,
                size=(50, 15),
                key="-DIAMETER SLIDER-",
                font=('Times New Roman', 10, 'bold'),
            ),
            sg.VerticalSeparator(),  # Separator 
            sg.Text("FOCAL LENGTH (inches) [DEFAULT: 48]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-MIRROR PARAMS B-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (1, 255),
                48,
                0.25,
                orientation="h",
                enable_events=True,
                size=(50, 15),
                key="-FOCAL LENGTH SLIDER-",
                font=('Times New Roman', 10, 'bold'),
            ),
            sg.VerticalSeparator(),  # Separator 
        ],
        [sg.Button("Exit", size=(10, 1)), sg.VerticalSeparator(), sg.Button("About", size=(10, 1)), sg.VerticalSeparator(), sg.Button("Save Image", size=(15, 1)) ],
    ]


    # Create the window
    window = sg.Window("FKESA v2 GUI [LIVE]", layout)

    # Start the thread for processing frames
    thread = threading.Thread(target=process_frames)
    thread.daemon = True
    thread.start()


    #thread = threading.Thread(target=update_gui, args=(window,))
    #thread.daemon = True
    #thread.start()

    while True:
        event, values = window.read(timeout=20)  # Update the GUI every 20 milliseconds
        if event == sg.WIN_CLOSED or event == 'Exit':
            processing_frames_running = False  # Signal the processing_frames thread to exit
            exit_event.set()  # Signal the processing_frames thread to exit
            break
        elif event == "-PAUSE PLAY VIDEO-":
             if is_playing == True:
                window['-PAUSE PLAY VIDEO-'].update(button_color = ('black','yellow'))
                window['-PAUSE PLAY VIDEO-'].update(text = ('PLAY VIDEO'))
                is_playing = False
             elif is_playing == False:
                window['-PAUSE PLAY VIDEO-'].update(button_color = ('white','green'))
                window['-PAUSE PLAY VIDEO-'].update(text = ('PAUSE VIDEO'))
                is_playing = True
        elif event == "-RAW VIDEO SELECT-":
             if values["-RAW VIDEO SELECT-"] == False:
                raw_video = False
             elif values["-RAW VIDEO SELECT-"] == True:
                raw_video = True
        elif event == "-COLOR VIDEO SELECT-":
             if values["-COLOR VIDEO SELECT-"] == False:
                color_video = False
             elif values["-COLOR VIDEO SELECT-"] == True:
                color_video = True
        elif event == "-DELAY SLIDER-":
             fkesa_time_delay = int(values["-DELAY SLIDER-"])
        # Inside the main event loop where the sliders are handled
        elif event == "-MINDIST SLIDER-" \
             or event == "-PARAM SLIDER A-" or event == "-PARAM SLIDER B-" \
             or event == "-RADIUS SLIDER A-" or event == "-RADIUS SLIDER B-" \
             or event == "-BRIGHTNESS SLIDER-" or event == "-ZONES SLIDER-" \
             or event == "-ANGLE SLIDER-" or event == "-SKIP ZONES SLIDER-" \
             or event == "-DIAMETER SLIDER-" or event == "-FOCAL LENGTH SLIDER-" :
            mindist_val = int(values["-MINDIST SLIDER-"])
            param_a_val = int(values["-PARAM SLIDER A-"])
            param_b_val = int(values["-PARAM SLIDER B-"])
            radius_a_val = int(values["-RADIUS SLIDER A-"])
            radius_b_val = int(values["-RADIUS SLIDER B-"])
            brightness_tolerance_val = int(values["-BRIGHTNESS SLIDER-"])
            zones_val = int(values["-ZONES SLIDER-"])
            angle_val = int(values["-ANGLE SLIDER-"])
            diameter_val = float(values["-DIAMETER SLIDER-"])
            focal_length_val = float(values["-FOCAL LENGTH SLIDER-"])
            skip_zones_val = int(values["-SKIP ZONES SLIDER-"])
        elif event == 'Save Image':
            with lock:
              if shared_frame is not None:
                # Use OpenCV to write the image data to a file
                filename = f"fkesa_v2_{int(time.time())}.png"  # Generate a filename (you can adjust this)
                cv2.imwrite(filename, shared_frame)
                #with open(filename, 'wb') as f:
                #    f.write(shared_frame)
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
              image.thumbnail((640, 480))  # Resize image if needed
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

        
        with lock:
          # Update the GUI from the main thread
          if 'shared_frame' in globals():
            if shared_frame is not None and window is not None:
               imgbytes = cv2.imencode('.png', shared_frame)[1].tobytes()
               window['-IMAGE-'].update(data=imgbytes)
       

    # Wait for the processing thread to complete before closing the window
    if thread.is_alive():
       thread.join()

    if cap is not None:
       cap.release()  # Release the camera explicitly
    window.close()

except Exception as e:
        print(f"An error occurred: {e}")

