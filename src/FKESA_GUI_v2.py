#!/usr/bin/env python3
#Author: Pratik M Tambe <enthusiasticgeek@gmail.com>
#Date: Dec 18, 2023

import math
import cv2
import PySimpleGUI as sg
#from PIL import Image, ImageTk
from PIL import Image as PILImage
import io
import numpy as np
import os.path
import time
import threading
from FKESA_v2_core import FKESABuilder 
from FKESA_v2_helper import FKESAHelper 
#Helper function is to be used with the following MCU code
#https://github.com/enthusiasticgeek/esp32_wroom_wifiap_stepper_webserver/blob/main/esp32_webserver_wifiap_stepper_control_pushbuttons_isr_xy_end_switches.ino
from datetime import datetime
import tempfile
import subprocess
import platform

import tkinter as tk

# Create a hidden tkinter root window
root = tk.Tk()
root.withdraw()

# Get screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
scale_window = False


# Initialize a variable to store image data
#image_data = None
process_fkesa = False
selected_camera=0
shared_frame = None
cap = None
# Create synchronization events
exit_event = threading.Event()  # Event to signal thread exit
is_playing = True
is_recording = False
is_measuring = False
is_auto = False

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
fkesa_time_delay = 300
current_time = time.time()
measurement_run_counter = 0
step_size_val = 0.010
step_delay_val = 50 #microsec
stepper_microsteps = 128
stepper_steps_per_revolution = 200

# Get the current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"fkesa_v2_{timestamp}.csv"
output_folder = f"fkesa_v2_{timestamp}_output"


# Get the user's home directory
home_dir = os.path.expanduser("~")
#Example Specify the relative path from the home directory
#image_path = os.path.join(home_dir, 'Desktop', 'fkesa_v2.bmp')
# Perform the image write operation
#if not cv2.imwrite(image_path, img2):
#    raise Exception("Could not write image")


#Define video recording parameters
# Define the codec and create VideoWriter object
# See https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#fourcc = cv2.VideoWriter_fourcc(*'X264')
#fourcc = cv2.VideoWriter_fourcc(*'WMV1')
#fourcc = cv2.VideoWriter_fourcc(*'WMV2')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,  480))
out = None

# Create the splash screen layout
splash_layout = [
    [sg.Text('Welcome to Foucault KnifeEdge Shadowgram Analyzer!', justification='center')],
    [sg.Text('FKESA GUI Version 2.1', justification='center')],
    [sg.Text('Author: Pratik M. Tambe <enthusiasticgeek@gmail.com>', justification='center')],
    [sg.Image('fkesa.png')],  
    [sg.Text('Launching...', justification='center',k='-LAUNCH-')],
    [sg.ProgressBar(100, orientation='h', s=(60,20), k='-PBAR-')]
]
# Create the splash screen window
splash_window = sg.Window('Splash Screen', splash_layout, no_titlebar=True, keep_on_top=True)
# Show splash screen for 3 seconds
start_time = time.time()
while time.time() - start_time < 3:
    event, values = splash_window.read(timeout=100)
    if (time.time() - start_time) < 1:
        progress = 33
        splash_window['-PBAR-'].update(progress)
        splash_window['-LAUNCH-'].update('Launching.', text_color='white')
    elif (time.time() - start_time) < 2:
        progress = 66
        splash_window['-PBAR-'].update(progress)
        splash_window['-LAUNCH-'].update('Launching..', text_color='white')
splash_window['-LAUNCH-'].update('Searching Available Camera Devices...Please allow a few seconds...', text_color='white')
progress = 100
splash_window['-PBAR-'].update(progress)
time.sleep(1)
splash_window.close()

# -------------------------------------------------------------------------
# ------------ Scaling ------------
def get_scaling():
    # called before window created
    root = sg.tk.Tk()
    scaling = root.winfo_fpixels('1i')/72
    root.destroy()
    return scaling

# Find the number in original screen when GUI designed.
my_scaling = 1.334646962233169      # call get_scaling()
my_width, my_height = 1920, 1080     # call sg.Window.get_screen_size()

# Get the number for new screen
scaling_old = get_scaling()
width, height = sg.Window.get_screen_size()

scaling = scaling_old * min(my_width / width, my_height / height)
#scaling = scaling_old * 0.75

if scale_window == True:
   sg.set_options(scaling=scaling)
# -------------------------------------------------------------------------



# Function to check if <another_file>.py is already running
def is_another_file_instance_running(file_lock):
    # Check for the existence of a lock file
    return os.path.exists(os.path.join(tempfile.gettempdir(), file_lock))

# Function to create a lock file
def create_lock_file(file_lock):
    lock_file_path = os.path.join(tempfile.gettempdir(), file_lock)
    with open(lock_file_path, 'w'):
        pass

# Function to remove the lock file
def remove_lock_file(file_lock):
    lock_file_path = os.path.join(tempfile.gettempdir(), file_lock)
    if os.path.exists(lock_file_path):
        os.remove(lock_file_path)

def calculate_fps(delay_ms):
    fps = float(1 / (delay_ms * 0.001))
    return fps

def is_valid_number(input_str):
    try:
        float(input_str)
        return True
    except ValueError:
        return False

def is_valid_integer(input_str):
    try:
        int(input_str)
        return True
    except ValueError:
        return False

def mm_to_inches(mm):
    return mm / 25.4

def inches_to_mm(inches):
    return inches * 25.4

def check_step_size_validity(values):
        step_size = values['step_size']
        if is_valid_number(step_size):
            step_size = format(float(step_size), '.3f')
            #print(f'Success! Step Size: {step_size}')
            return True
        else:
            return False
            #sg.popup_error('Invalid input! Please enter a valid integer or floating-point number.')


def author_window():
    layout = [

        [sg.Image(filename='fkesa.ico.png'),],
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
    camera = None
    while len(non_working_ports) < 2: # if there are more than 2 non working ports stop the testing. 
        if platform.system() == "Linux":
           camera = cv2.VideoCapture(dev_port)
        elif platform.system() == "Windows":
           camera = cv2.VideoCapture(dev_port, cv2.CAP_DSHOW)
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
    global csv_filename
    global processing_frames_running
    global selected_camera
    global cap
    global exit_event
    global current_time
    global fourcc
    global out
    # counter to ease CPU processing with modulo operator
    counter=0
    if cap is not None:
        cap.release()
    if platform.system() == "Linux":
       cap = cv2.VideoCapture(selected_camera)
    elif platform.system() == "Windows":
       cap = cv2.VideoCapture(selected_camera, cv2.CAP_DSHOW)
    #cap = cv2.VideoCapture(selected_camera)  # Open the default camera
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

                global output_folder
                global measurement_run_counter
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
                global is_playing
                global is_recording
                global is_measuring
                global is_auto
                global step_size_val
                global step_delay_val
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
                   if counter % 5 == 0:
                        if counter > 100:
                           counter = 0
                        """
                        start_time = time.time()
                        """
                        builder = FKESABuilder()
                        builder.with_folder(output_folder+'_run_'+str(measurement_run_counter))
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
                        builder.with_param('csv_filename', csv_filename)
                        builder.with_param('append_to_csv', is_measuring or is_auto)
                        builder.with_param('step_size', step_size_val)
                        # ... Include other parameters as needed

                        # Build and execute the operation
                        _,fkesa_frame = builder.build(frame)
                        time.sleep(fkesa_time_delay / 1000)
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
                   else:
                        fkesa_frame = preprocess_frame

                if fkesa_frame is None:
                   continue

                global shared_frame
                if fkesa_frame is not None:
                   if is_auto:
                      is_auto=False
                   shared_frame = fkesa_frame.copy()
                   #Record if flag set
                   if is_recording == True:
                      if out is not None:
                         print("Continuing Video Recording....")
                         if raw_video==True:
                            shared_frame = cv2.resize(shared_frame, (640, 480))
                         else:
                            #Video reording writer needs RGB or it silently fails (flaw in OpenCV ver 2)
                            shared_frame = cv2.cvtColor(shared_frame, cv2.COLOR_BGRA2RGB)
                         out.write(shared_frame)
                   #elif is_recording == False:
                   #   if out is not None:
                   #      print("Stopping Video Recording....")
                   #      out.release()
                   #      out = None
        except Exception as e:
               print(f"An exception occurred {e}")
      

    cap.release()
    cv2.destroyAllWindows()


#================= Stepper motor distance conversion ================
def inches_to_steps(distance_inches, steps_per_revolution, microsteps=1):
    # Calculate steps based on inches, steps per revolution, and microsteps
    steps = distance_inches * steps_per_revolution * microsteps / (1.8 * math.pi)  # 1.8 is the stepper motor's step angle in degrees
    return int(steps)  # Return the number of steps as an integer

def distance_to_steps(distance_mm, steps_per_rev, microstepping_factor, ball_screw_pitch_mm):
    # Calculate the total number of steps per revolution, considering microstepping
    total_steps_per_rev = steps_per_rev * microstepping_factor
    # Calculate the distance traveled in one revolution of the ball screw
    distance_per_rev = ball_screw_pitch_mm
    # Calculate the total number of steps needed to move the specified distance
    steps = int((distance_mm / distance_per_rev) * total_steps_per_rev)
    return steps


"""
# Example usage:
steps_per_revolution = 200  # Replace with your stepper motor's steps per revolution
microsteps = 100  # Replace with your microstepping value
distance_inches = 0.010  # Replace with the distance in inches you want to convert

result_steps = inches_to_steps(distance_inches, steps_per_revolution, microsteps)
print(f"{distance_inches} inches is approximately {result_steps} steps.")

# Example usage:
distance_to_travel = 100  # Specify the distance in mm
steps_per_revolution = 200  # Specify the number of steps per revolution of your stepper motor
microstepping_factor = 32  # Specify the microstepping factor of your stepper driver
ball_screw_pitch = 5  # Specify the pitch of the ball screw in mm

steps_needed = distance_to_steps(distance_to_travel, steps_per_revolution, microstepping_factor, ball_screw_pitch)
print(f"Distance to travel: {distance_to_travel} mm")
print(f"Steps needed: {steps_needed} steps")


"""
#================= Stepper motor distance conversion ================

#================= Auto-Foucault process ================

autofoucault_error_lut = {
    1: "Cannot set CW steps (X)",
    2: "Cannot set CW delay (X)",
    3: "Cannot find start position - limit switch (X) ",
    4: "Max attempts to find start position - limit switch reached (X)",
    5: "Cannot reset counters",
    6: "Cannot find end position - limit switch (X) ",
    7: "Max attempts to find end position - limit switch reached (X)"
}

def process_fkesa_v2(device_ip="192.168.4.1", result_delay_usec=50, result_steps=500, max_attempts=50):
        global exit_event
        global thread
        #Default IP 192.168.4.1
        helper = FKESAHelper()
        #helper_attempts = 0
        #max_attempts = 50
        found_start_x = False
        found_end_x = False

        #result_steps = inches_to_steps(distance_inches, steps_per_revolution, microsteps)
        try:
                # Steps
                url_post = f"http://{device_ip}/button1"
                data_post = {"textbox1": f"{result_steps}"}
                # Call the post_data method on the instance
                response_post = helper.post_data_to_url(url_post, data_post)
                if response_post is not None:
                    print("POST Request Response:")
                    print(response_post.text)
                    print("Headers:")
                    print(response_post.headers)
                else:
                    print("no response!")
                    return False, 1
                # Microsecs
                url_post = f"http://{device_ip}/button2"
                data_post = {"textbox2": f"{result_delay_usec}"}
                # Call the post_data method on the instance
                response_post = helper.post_data_to_url(url_post, data_post)
                if response_post is not None:
                    print("POST Request Response:")
                    print(response_post.text)
                    print("Headers:")
                    print(response_post.headers)
                else:
                    print("no response!")
                    return False, 2
                for num in range(0, max_attempts):
                        url_boolean = f"http://{device_ip}/reached_begin_x"
                        # Call the boolean method on the instance
                        result_boolean = helper.get_boolean_value_from_url(url_boolean)
                        if result_boolean is not None:
                            print("Received boolean value:", result_boolean)
                            if result_boolean:
                               found_start_x = True
                               break
                            elif not result_boolean:
                                # CCW X
                                url_post = f"http://{device_ip}/button4"
                                data_post = None
                                # Call the post_data method on the instance
                                response_post = helper.post_data_to_url(url_post, data_post)
                                if response_post is not None:
                                    print("POST Request Response:")
                                    print(response_post.text)
                                    print("Headers:")
                                    print(response_post.headers)
                                else:
                                    print("no response!")
                                    return False, 3
                        #Allow some time for carriage to move along the stepper motor rail
                        time.sleep(1)
                if not found_start_x:
                   #raise ValueError('ERROR: Could not find start reference!')
                   print('ERROR: Max attempts reached! Could not find start reference [Auto-Foucault]!')
                   return False, 4

                # Reset Counters
                url_string = f"http://{device_ip}/reset"
                result_string = helper.get_string_value_from_url(url_string)
                if result_string is not None:
                   print("Received string value:", result_string)
                else:
                   print("no response!")
                   return False, 5

                # FKESA v2 process begins below now we have carriage at the start of the rails.

                for num in range(0, max_attempts):
                        url_boolean = f"http://{device_ip}/reached_end_x"
                        # Call the boolean method on the instance
                        result_boolean = helper.get_boolean_value_from_url(url_boolean)
                        if result_boolean is not None:
                            print("Received boolean value:", result_boolean)
                            if result_boolean:
                               found_end_x = True
                               break
                            elif not result_boolean:
                                # CW X
                                url_post = f"http://{device_ip}/button3"
                                data_post = None
                                # Call the post_data method on the instance
                                response_post = helper.post_data_to_url(url_post, data_post)
                                if response_post is not None:
                                    print("POST Request Response:")
                                    print(response_post.text)
                                    print("Headers:")
                                    print(response_post.headers)
                                else:
                                    print("no response!")
                                    return False, 6
                            # ====== FKESA v2 process iteration begin =========
                            with lock:
                                    global is_playing
                                    global is_recording
                                    global is_measuring
                                    global is_auto
                                    is_playing = False
                                    is_recording = False
                                    is_measuring = False
                                    is_auto = False
                                    exit_event.set()  # Set the exit event to stop the loop
                            # Wait for the processing thread to complete before closing the window
                            if thread.is_alive():
                               thread.join()
                            #Some time to stop and resume
                            time.sleep(1)
                            # Resume the worker thread
                            print("Resuming the worker thread...")
                            exit_event.clear()  # Clear the exit event to allow the loop to continue
                            #Get to the initial state of boolean values
                            with lock:
                                    #global is_playing
                                    #global is_recording
                                    #global is_measuring
                                    #global is_auto
                                    is_playing = True
                                    is_recording = False
                                    is_measuring = False
                                    is_auto = True
                            # Start the thread for processing frames
                            thread = threading.Thread(target=process_frames)
                            thread.daemon = True
                            thread.start()
                            # ====== FKESA v2 process iteration end =========
                        #Allow some time for carriage to move along the stepper motor rail
                        time.sleep(1)
                if not found_end_x:
                   #raise ValueError('ERROR: Could not find end reference!')
                   print('ERROR: Max attempts reached! Could not find end reference [Auto-Foucault]!')
                   return False, 7
        except Exception as e:
               print(str(e))
        return True

#process_fkesa_v2("192.168.4.1",50,100,10)

#=========== main ==========


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
        [sg.Text("SELECT IMAGES FOLDER",font=('Times New Roman', 10, 'bold'), text_color="darkred"),], 
        [
            #sg.Text("Select Images Folder",font=('Times New Roman', 12, 'bold'), text_color="darkred"), 
            sg.In(size=(30, 1), enable_events=True, key="-FOLDER-"),
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
        [sg.Text("CLICK TO VIEW AN IMAGE FROM THE LEFT PANE", font=('Times New Roman', 10, 'bold'), text_color="darkred")],
        [sg.Text(size=(70, 1), key="-TOUT-")],
        [sg.Image(key="-LOAD IMAGE-",size=(200,200))],
    ]


    # Define the window layout
    layout = [
        [sg.Image(filename='fkesa.ico.png'), sg.Text("FOUCAULT KNIFE-EDGE SHADOWGRAM ANALYZER (FKESA) VERSION 2", size=(100, 1), justification="left", font=('Times New Roman', 10, 'bold'),text_color='darkgreen')],
        [sg.Menu(menu_def, background_color='lightblue',text_color='navy', disabled_text_color='yellow', font='Verdana', pad=(10,10))],
        [sg.HorizontalSeparator()],  # Separator 
        [sg.Image(filename="", key="-IMAGE-", size=(640,480)), sg.VerticalSeparator(), sg.Column(file_list_column), sg.VerticalSeparator(), sg.Column(image_viewer_column),],
        [
            [
             sg.Button('Start Recording', key='-RECORD VIDEO-',button_color = ('white','red')), 
             sg.VerticalSeparator(), 
             sg.Button('Pause Video', key='-PAUSE PLAY VIDEO-',button_color = ('white','green')) , 
             sg.VerticalSeparator(), 
             sg.Button("Save Image", size=(15, 1), button_color = ('white','blue')), 
             sg.VerticalSeparator(), 
             sg.Text("STEP SIZE (INCHES)", size=(18, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-STEP SIZE-"),
             sg.InputText('0.010', key='step_size', size=(10, 1), enable_events=True, justification='center', tooltip='Enter an integer or floating-point number'),
             sg.VerticalSeparator(),  # Separator 
             sg.Text("STEP DELAY (μSECS)", size=(20, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-PULSE DELAY-"),
             sg.InputText('50', key='step_delay', size=(10, 1), enable_events=True, justification='center', tooltip='Enter an integer number'),
             sg.VerticalSeparator(),  # Separator 
             sg.Button('Auto Foucault', key='-AUTOFOUCAULT-',button_color = ('black','violet'),disabled=True),
             sg.VerticalSeparator(),  # Separator 
             sg.Button('Start Measurements', key='-MEASUREMENTS-',button_color = ('black','orange'),disabled=True),
             sg.VerticalSeparator(),  # Separator 
             sg.Button('View Measurements Data', key='-MEASUREMENTS CSV-',button_color = ('white','black'),disabled=False),
             sg.VerticalSeparator(),  # Separator 
             sg.Button('Optical Ray Diagram', key='-OPTICS-',button_color = ('white','brown'),disabled=False),
             sg.VerticalSeparator(),  # Separator 
            ],
            [sg.HorizontalSeparator()],  # Separator 
            #[sg.DropDown(working_ports, default_value='0', enable_events=True, key='-CAMERA SELECT-')],
            [
             sg.DropDown(working_ports, default_value='0', enable_events=True, key='-CAMERA SELECT-', background_color='green', text_color='white'), 
             sg.Button('SELECT CAMERA'), 
             sg.VerticalSeparator(), 
             sg.Checkbox('RAW VIDEO', default=True, enable_events=True, key='-RAW VIDEO SELECT-',font=('Times New Roman', 10, 'bold')), 
             sg.VerticalSeparator(), 
             sg.Checkbox('COLORED RAW VIDEO', default=True, enable_events=True, key='-COLOR VIDEO SELECT-', font=('Times New Roman', 10, 'bold')), 
             sg.VerticalSeparator(),  # Separator 
            ],
            #[sg.Button('SELECT CAMERA'), sg.VerticalSeparator(), sg.Button('Cancel'), sg.VerticalSeparator()], 
        ],
        [sg.HorizontalSeparator()],  # Separator 
        [sg.Text("CIRCULAR HOUGH TRANSFORM PARAMETERS [MIRROR DETECTION]", size=(60, 1), justification="left", font=('Times New Roman', 10, 'bold'), text_color='darkred')],
        [sg.HorizontalSeparator()],  # Separator 
        [
            sg.Text("MIN DIST (PIXELS) [DEFAULT: 50]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-MINDIST A-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (1, 255),
                50,
                1,
                orientation="h",
                enable_events=True,
                size=(50, 10),
                key="-MINDIST SLIDER-",
                font=('Times New Roman', 10, 'normal'),
                # text_color=('darkgreen') # experimental
            ),
            sg.VerticalSeparator(),  # Separator 
            sg.Text("PROCESSING DELAY MILLISECONDS [DEFAULT: 100]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-MINDIST B-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (0,1000),
                300,
                100,
                orientation="h",
                enable_events=True,
                size=(50, 10),
                key="-DELAY SLIDER-",
                font=('Times New Roman', 10, 'normal'),
            ),
            sg.VerticalSeparator(),  # Separator 
        ],
        [
            sg.Text("PARAMETERS 1 (PIXELS) [DEFAULT: 25]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-PARAMS A-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (1, 255),
                25,
                1,
                orientation="h",
                enable_events=True,
                size=(50, 10),
                key="-PARAM SLIDER A-",
                font=('Times New Roman', 10, 'normal'),
            ),
            sg.VerticalSeparator(),  # Separator 
            sg.Text("PARAMETER 2 (PIXELS) [DEFAULT: 60]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-PARAMS B-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (1, 255),
                60,
                1,
                orientation="h",
                enable_events=True,
                size=(50, 10),
                key="-PARAM SLIDER B-",
                font=('Times New Roman', 10, 'normal'),
            ),
            sg.VerticalSeparator(),  # Separator 
        ],
        [
            sg.Text("MIN RADIUS (PIXELS) [DEFAULT: 10]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-MIN RADIUS-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (1, 255),
                20,
                1,
                orientation="h",
                enable_events=True,
                size=(50, 10),
                key="-RADIUS SLIDER A-",
                font=('Times New Roman', 10, 'normal'),
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
                size=(50, 10),
                key="-RADIUS SLIDER B-",
                font=('Times New Roman', 10, 'normal'),
            ),
            sg.VerticalSeparator(),  # Separator 
        ],
        [sg.HorizontalSeparator()],  # Separator 
        [sg.Text("INTENSITY PARAMETERS [NULL ZONES IDENTIFICATION]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), text_color='darkred')],
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
                size=(50, 10),
                key="-BRIGHTNESS SLIDER-",
                font=('Times New Roman', 10, 'normal'),
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
                size=(50, 10),
                key="-ZONES SLIDER-",
                font=('Times New Roman', 10, 'normal'),
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
                size=(50, 10),
                enable_events=True,
                key="-ANGLE SLIDER-",
                font=('Times New Roman', 10, 'normal'),
            ),
            sg.VerticalSeparator(),  # Separator 
            sg.Text("SKIP ZONES FROM THE MIRROR CENTER [DEFAULT: 10]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-SKIP ZONES-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (1, 20),
                10,
                1,
                orientation="h",
                size=(50, 10),
                enable_events=True,
                key="-SKIP ZONES SLIDER-",
                font=('Times New Roman', 10, 'normal'),
            ),
            sg.VerticalSeparator(),  # Separator 
        ],
        [sg.HorizontalSeparator()],  # Separator 
        [sg.Text("PRIMARY MIRROR PARAMETERS [PARABOLIC MIRROR or K = -1]", size=(60, 1), justification="left", font=('Times New Roman', 10, 'bold'), text_color='darkred')],
        [sg.HorizontalSeparator()],  # Separator 
        [
            sg.Text("DIAMETER (INCHES) [DEFAULT: 6]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-MIRROR PARAMS A-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (1, 255),
                6,
                0.25,
                orientation="h",
                enable_events=True,
                size=(50, 10),
                key="-DIAMETER SLIDER-",
                font=('Times New Roman', 10, 'normal'),
            ),
            sg.VerticalSeparator(),  # Separator 
            sg.Text("FOCAL LENGTH (INCHES) [DEFAULT: 48]", size=(50, 1), justification="left", font=('Times New Roman', 10, 'bold'), key="-MIRROR PARAMS B-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (1, 255),
                48,
                0.25,
                orientation="h",
                enable_events=True,
                size=(50, 10),
                key="-FOCAL LENGTH SLIDER-",
                font=('Times New Roman', 10, 'normal'),
            ),
            sg.VerticalSeparator(),  # Separator 
        ],
        [sg.Button("Exit", size=(10, 1), button_color=('white','darkred')), sg.VerticalSeparator(), sg.Button("About", size=(10, 1)), sg.VerticalSeparator(), ],
    ]


    # Create the window
    #window = sg.Window("FKESA v2 GUI [LIVE]", layout, size=(1640, 1040), resizable=True)
    window = sg.Window("FKESA v2 GUI", layout, size=(screen_width, screen_height), resizable=True)

    # Start the thread for processing frames
    thread = threading.Thread(target=process_frames)
    thread.daemon = True
    thread.start()


    #thread = threading.Thread(target=update_gui, args=(window,))
    #thread.daemon = True
    #thread.start()

    while True:
        event, values = window.read(timeout=20)  # Update the GUI every 20 milliseconds
        if event == 'Exit':
           confirm_exit = sg.popup_yes_no("Are you sure you want to exit?")
           if confirm_exit == "Yes":
              processing_frames_running = False  # Signal the processing_frames thread to exit
              exit_event.set()  # Signal the processing_frames thread to exit
              break
        elif event == sg.WIN_CLOSED:
              processing_frames_running = False  # Signal the processing_frames thread to exit
              exit_event.set()  # Signal the processing_frames thread to exit
              break
        elif event == "-AUTOFOUCAULT-":
           confirm_proceed = sg.popup_yes_no("Is your Foucault setup ready? Are you sure you want to proceed with AutoFoucault?\n\nNote that this test may take a few minutes to complete. You will not be able to use other widgets on this GUI during this operation.")
           if confirm_proceed == "Yes":
              with lock:
                      # First stop any ongoing measurements
                      if is_measuring == True:
                        print("Stopping measurements.....") 
                        window['-MEASUREMENTS-'].update(button_color = ('black','orange'))
                        window['-MEASUREMENTS-'].update(text = ('Start Measurements'))
                        window['-DIAMETER SLIDER-'].update(disabled=False)
                        window['-FOCAL LENGTH SLIDER-'].update(disabled=False)
                        window['step_size'].update(disabled=False)
                        window['step_delay'].update(disabled=False)
                        is_measuring = False
                      # Disable all Widgets temporarily
                      window['-DIAMETER SLIDER-'].update(disabled=True)
                      window['-FOCAL LENGTH SLIDER-'].update(disabled=True)
                      window['-MEASUREMENTS-'].update(disabled=True)
                      window['-MEASUREMENTS CSV-'].update(disabled=True)
                      window['step_size'].update(disabled=True)
                      window['step_delay'].update(disabled=True)
                      window['-PAUSE PLAY VIDEO-'].update(disabled=True)
                      window['-RECORD VIDEO-'].update(disabled=True)
                      window['-RAW VIDEO SELECT-'].update(disabled=True)
                      window['-COLOR VIDEO SELECT-'].update(disabled=True)
                      window['-CAMERA SELECT-'].update(disabled=True)

              distance_inches=float(values['step_size'])
              #result_steps = inches_to_steps(distance_inches, stepper_steps_per_revolution, stepper_microsteps)
              ball_screw_pitch_mm = 5
              distance_mm = inches_to_mm(distance_inches)
              result_steps = distance_to_steps(distance_mm, stepper_steps_per_revolution, stepper_microsteps, ball_screw_pitch_mm)
              result_delay_usec = values['step_delay']
              success, error = process_fkesa_v2(device_ip="192.168.4.1", result_delay_usec=result_delay_usec, result_steps=result_steps, max_attempts=50)
              if not success:
                    sg.popup_ok(f"FKESA AUTOFOUCAULT Failed with an error # {error} -> \"{autofoucault_error_lut[error]}\". Click OK to continue.") 
              with lock:
                      # Re-enable all Widgets
                      window['-DIAMETER SLIDER-'].update(disabled=False)
                      window['-FOCAL LENGTH SLIDER-'].update(disabled=False)
                      window['-MEASUREMENTS-'].update(disabled=False)
                      window['-MEASUREMENTS CSV-'].update(disabled=False)
                      window['step_size'].update(disabled=False)
                      window['step_delay'].update(disabled=False)
                      window['-PAUSE PLAY VIDEO-'].update(disabled=False)
                      window['-RECORD VIDEO-'].update(disabled=False)
                      window['-RAW VIDEO SELECT-'].update(disabled=False)
                      window['-COLOR VIDEO SELECT-'].update(disabled=False)
                      window['-CAMERA SELECT-'].update(disabled=False)
        elif event == "-MEASUREMENTS-":
            with lock:
             if is_measuring == False:
                if check_step_size_validity(values) == True:
                   print("Starting measurements.....") 
                   window['-MEASUREMENTS-'].update(button_color = ('orange','black'))
                   window['-MEASUREMENTS-'].update(text = ('Stop Measurements'))
                   window['-DIAMETER SLIDER-'].update(disabled=True)
                   window['-FOCAL LENGTH SLIDER-'].update(disabled=True)
                   window['-AUTOFOUCAULT-'].update(disabled=True)
                   step_size_val = float(values['step_size'])
                   step_delay_val = float(values['step_delay'])
                   window['step_size'].update(disabled=True)
                   window['step_delay'].update(disabled=True)
                   measurement_run_counter+=1
                   is_measuring = True
                else:
                   sg.popup_error('Invalid input! Please enter a valid integer or floating-point number.')
             elif is_measuring == True:
                print("Stopping measurements.....") 
                window['-MEASUREMENTS-'].update(button_color = ('black','orange'))
                window['-MEASUREMENTS-'].update(text = ('Start Measurements'))
                window['-DIAMETER SLIDER-'].update(disabled=False)
                window['-FOCAL LENGTH SLIDER-'].update(disabled=False)
                window['-AUTOFOUCAULT-'].update(disabled=False)
                window['step_size'].update(disabled=False)
                window['step_delay'].update(disabled=False)
                is_measuring = False
        elif event == "-MEASUREMENTS CSV-":
             if not is_another_file_instance_running('measurement_csv'):
                try:
                    create_lock_file('measurement_csv')  # Create lock file
                    window['-MEASUREMENTS CSV-'].update(disabled=True, text='Viewing Measurements Data')
                    subprocess.run(['python', './FKESA_v2_csv.py'])
                except Exception as e:
                    sg.popup_error(f"Error running /FKESA_v2_csv.py: {e}")
                finally:
                    remove_lock_file('measurement_csv')  # Remove lock file
                    window['-MEASUREMENTS CSV-'].update(disabled=False, text='View Measurements Data')
        elif event == "-OPTICS-":
             if not is_another_file_instance_running('optics'):
                try:
                    create_lock_file('optics')  # Create lock file
                    window['-OPTICS-'].update(disabled=True, text='Optical Ray Diagram')
                    subprocess.run(['python', './FKESA_v2_optics.py', '--radius', str(diameter_val/2), '--focal_length', str(focal_length_val)])
                except Exception as e:
                    sg.popup_error(f"Error running /FKESA_v2_optics.py: {e}")
                finally:
                    remove_lock_file('optics')  # Remove lock file
                    window['-OPTICS-'].update(disabled=False, text='Optical Ray Diagram')

        elif event == "-RECORD VIDEO-":
             if is_recording == False:
                print("Starting video recording.....") 
                window['-RECORD VIDEO-'].update(button_color = ('black','yellow'))
                window['-RECORD VIDEO-'].update(text = ('Stop Recording'))
                video_filename = f"fkesa_v2_{int(time.time())}.avi"  # Generate a filename (you can adjust this)
                if out is not None:
                   out.release()
                   out = None
                if raw_video == True:
                   out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640,  480))
                else:
                   #out = cv2.VideoWriter(video_filename, fourcc, 1.0, (640,  480))
                   fps = calculate_fps(fkesa_time_delay)
                   out = cv2.VideoWriter(video_filename, fourcc, float(fps), (640,  480))
                is_recording = True
             elif is_recording == True:
                print("Stopping video recording.....") 
                window['-RECORD VIDEO-'].update(button_color = ('white','red'))
                window['-RECORD VIDEO-'].update(text = ('Start Recording'))
                if out is not None:
                   out.release()
                   out = None
                is_recording = False
        elif event == "-PAUSE PLAY VIDEO-":
             if is_playing == True:
                window['-PAUSE PLAY VIDEO-'].update(button_color = ('black','yellow'))
                window['-PAUSE PLAY VIDEO-'].update(text = ('Play Video'))
                is_playing = False
                print("Pausing the worker thread...")
                exit_event.set()  # Set the exit event to stop the loop
                # Wait for the processing thread to complete before closing the window
                if thread.is_alive():
                   thread.join()
             elif is_playing == False:
                window['-PAUSE PLAY VIDEO-'].update(button_color = ('white','green'))
                window['-PAUSE PLAY VIDEO-'].update(text = ('Pause Video'))
                is_playing = True
                # Resume the worker thread
                print("Resuming the worker thread...")
                exit_event.clear()  # Clear the exit event to allow the loop to continue
                # Start the thread for processing frames
                thread = threading.Thread(target=process_frames)
                thread.daemon = True
                thread.start()
        elif event == "-RAW VIDEO SELECT-":
             if values["-RAW VIDEO SELECT-"] == False:
                window['-MEASUREMENTS-'].update(disabled=False)
                window['-AUTOFOUCAULT-'].update(disabled=False)
                raw_video = False
             elif values["-RAW VIDEO SELECT-"] == True:
                window['-MEASUREMENTS-'].update(disabled=True)
                window['-AUTOFOUCAULT-'].update(disabled=True)
                raw_video = True
                if is_measuring == True:
                  window['-MEASUREMENTS-'].update(button_color = ('black','orange'))
                  window['-MEASUREMENTS-'].update(text = ('Start Measuring'))
                  window['-DIAMETER SLIDER-'].update(disabled=False)
                  window['-FOCAL LENGTH SLIDER-'].update(disabled=False)
                  window['step_size'].update(disabled=False)
                  window['step_delay'].update(disabled=False)
                  is_measuring = False
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
            with lock:
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

                if platform.system() == "Linux":
                   if not cv2.imwrite(filename, shared_frame):
                      raise Exception("Could not write/save image")
                elif platform.system() == "Windows":
                   image_path = os.path.join(home_dir, 'Desktop', filename)
                   if not cv2.imwrite(image_path, shared_frame):
                      raise Exception("Could not write/save image")
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
              """
              image = Image.open(filename)
              image.thumbnail((640, 480))  # Resize image if needed
              bio = ImageTk.PhotoImage(image)
              window["-LOAD IMAGE-"].update(data=bio)
              """
              # Load the image and resize it to fit within 150x150 pixels
              image = PILImage.open(filename)
              image.thumbnail((533, 400))
              # Convert the resized image to bytes for PySimpleGUI
              bio = io.BytesIO()
              image.save(bio, format="PNG")
              window["-LOAD IMAGE-"].update(data=bio.getvalue())
            except Exception as e:
              print(e)
            except:
                pass
        elif event == 'About':
            #window.hide()  # Hide the main window
            author_window()  # Open the author information window
        elif event == 'SELECT CAMERA':
            if is_measuring == True:
                    sg.popup_ok("Please stop all measurements before switching cameras. Click OK to continue.") 
            elif is_measuring == False:
                    selected_camera = values['-CAMERA SELECT-']
                    print(f"Camera selected: {selected_camera}")
                    print("Stopping the worker thread...")
                    with lock:
                            is_playing = False
                            is_recording = False
                            is_measuring = False
                            is_auto = False
                    exit_event.set()  # Set the exit event to stop the loop
                    # Wait for the processing thread to complete before closing the window
                    if thread.is_alive():
                       thread.join()
                    #Some time to stop and resume
                    time.sleep(1)
                    # Resume the worker thread
                    print("Resuming the worker thread...")
                    exit_event.clear()  # Clear the exit event to allow the loop to continue
                    with lock:
                    #Get to the initial state of boolean values
                            is_playing = True
                            is_recording = False
                            is_measuring = False
                            is_auto = False
                    # Start the thread for processing frames
                    thread = threading.Thread(target=process_frames)
                    thread.daemon = True
                    thread.start()
        
        with lock:
          # Update the GUI from the main thread
          if 'shared_frame' in globals():
            if shared_frame is not None and window is not None:
               imgbytes = cv2.imencode('.png', shared_frame)[1].tobytes()
               window['-IMAGE-'].update(data=imgbytes)
        # Update the input background color based on validity
        input_background_color = 'white' if is_valid_number(values['step_size']) else 'pink'
        window['step_size'].update(background_color=input_background_color)
        input_background_color = 'white' if is_valid_integer(values['step_delay']) else 'pink'
        window['step_delay'].update(background_color=input_background_color)
 

    # Wait for the processing thread to complete before closing the window
    if thread.is_alive():
       thread.join()

    if cap is not None:
       cap.release()  # Release the camera explicitly

    if out is not None:
       out.release()

    window.close()

except Exception as e:
        print(f"An error occurred: {e}")

