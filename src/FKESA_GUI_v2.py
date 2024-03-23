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
import sys

import tkinter as tk

# Create a hidden tkinter root window
root = tk.Tk()
root.withdraw()

# Get screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
scale_window = False

is_debugging = False
autofoucault_simple_simulation = False
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
grids = False
auto_return = False
auto_error = -1

have_splash_screen=False

#Draw box
start_point = None
end_point = None

#3 point
point1 = None
point2 = None
point3 = None

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
fkesa_time_delay = 1000
current_time = time.time()
measurement_run_counter = 0
step_size_val = 0.10
step_delay_val = 50 #microsec
max_attempts_val = 10 #steps to traverse in autofoucault
stepper_microsteps = 32
stepper_steps_per_revolution = 200

autosave = False

#To use or not use circular hough transform
use_circular_hough_transform = False

step_counter = 0
prev_step_counter = 0

# Get the current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"fkesa_v2_{timestamp}.csv"
output_folder = f"fkesa_v2_{timestamp}_output"

distance_inches=0.0
#This is the pitch of stepper assembly - e.g. 1 full rotation of stepper (e.g. 200 steps) moves the carriage distance of Foucault Assembly by 5 mm.
ball_screw_pitch_mm=5
distance_mm = 10
result_steps = 50
result_delay_usec = 50

#thread
auto_thread = None

# Get the user's home directory
home_dir = os.path.expanduser("~")
print(home_dir)
#sys.exit(1)
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
if have_splash_screen:
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

if scale_window:
   sg.set_options(scaling=scaling)
# -------------------------------------------------------------------------

def convert_coordinates(x, y, canvas_height):
    new_y = canvas_height - y
    return x, new_y

#Ref: https://www.johndcook.com/blog/2023/06/18/circle-through-three-points/  
def circle_thru_pts(x1, y1, x2, y2, x3, y3):
    s1 = x1**2 + y1**2
    s2 = x2**2 + y2**2
    s3 = x3**2 + y3**2
    M11 = x1*y2 + x2*y3 + x3*y1 - (x2*y1 + x3*y2 + x1*y3)
    M12 = s1*y2 + s2*y3 + s3*y1 - (s2*y1 + s3*y2 + s1*y3)
    M13 = s1*x2 + s2*x3 + s3*x1 - (s2*x1 + s3*x2 + s1*x3)
    x0 =  0.5*M12/M11
    y0 = -0.5*M13/M11
    r0 = ((x1 - x0)**2 + (y1 - y0)**2)**0.5
    return (x0, y0, r0)

def bounding_box_from_circle(center_x, center_y, radius):
    start_x = center_x - radius
    start_y = center_y - radius
    end_x = center_x + radius
    end_y = center_y + radius
    return (int(start_x), int(480-start_y)), (int(end_x), int(480-end_y))
 

points = []

#Use this function with lock:
def draw_circle_thru_3_pts(canvas, points):
    global start_point
    global end_point
    #canvas.erase()
    if len(points) == 3:
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        # Calculate circle parameters
        x0, y0, r0 = circle_thru_pts(x1, y1, x2, y2, x3, y3)
        # Draw the circle
        canvas.draw_circle((x0, y0), r0, line_color='red', line_width=2)
        start_point, end_point = bounding_box_from_circle(x0, y0, r0)
        print("start ",start_point)
        print("end ",end_point)

def draw_rectangle(canvas, start_point, end_point):
    canvas.erase()
    canvas.draw_rectangle(start_point, end_point, line_color='red', line_width=2)

def draw_square(canvas, start_point, end_point):
    #canvas.erase()
    x1, y1 = start_point
    x2, y2 = end_point
    # Calculate the width and height of the rectangle
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    # Ensure that the width and height are equal
    size = min(width, height)
    # Adjust the end point to maintain the square shape
    if x2 < x1:
        x2 = x1 - size
    else:
        x2 = x1 + size
    if y2 < y1:
        y2 = y1 - size
    else:
        y2 = y1 + size
    # Draw the square
    canvas.draw_rectangle(start_point, (x2, y2), line_color='red')

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

def is_valid_mirror_params(value):
    output = is_valid_number(value)
    if output:
       if float(value) < 1.0 or float(value) > 255.0:
          return False
       else:
          return True
    return False
    


def mm_to_inches(mm):
    return mm / 25.4

def inches_to_mm(inches):
    return inches * 25.4

def check_step_size_validity(values):
        step_size = values['-STEP SIZE-']
        if is_valid_number(step_size):
            step_size = format(float(step_size), '.3f')
            #print(f'Success! Step Size: {step_size}')
            return True
        else:
            return False
            #sg.popup_error('Invalid input! Please enter a valid integer or floating-point number.')

def draw_mesh_grid(graph, w, h, color):
    #draw_graph(graph)
    for y in range(0,48):
        graph.DrawLine((0,y*h),(640,y*h), color=color, width=1)
    #draw_graph(graph)
    for x in range(0,64):
        graph.DrawLine((x*w,0),(x*w,480), color=color, width=1)

def author_window():
    layout = [

        [sg.Image(filename='fkesa.ico.png'),],
        [sg.Text("Foucault KnifeEdge Shadowgram Analyzer (FKESA) Version 2", size=(60, 1), justification="center", font=('Verdana', 10, 'bold'), key="-APP-")],
        [sg.HorizontalSeparator()],  # Separator 
        [sg.Text("Author: ", size=(8, 1), justification="left", font=('Verdana', 10, 'bold'), key="-AUTHOR-"),sg.Text('Pratik M. Tambe <enthusiasticgeek@gmail.com>')],
        [sg.Text("FKESA: ", size=(8, 1), justification="left", font=('Verdana', 10, 'bold'), key="-VERSION-"),sg.Text(' Version 2.1')],
        [sg.Text("Release Date: ", size=(14, 1), justification="left", font=('Verdana', 10, 'bold'), key="-RELEASE DATE-"),sg.Text('December 25, 2023')],
        [sg.Text("Credits/Feedback: ", size=(18, 1), justification="left", font=('Verdana', 10, 'bold'), key="-AUTHOR-")],
        [sg.Text('Guy Brandenburg, Alan Tarica - National Capital Astronomers (NCA)')],
        [sg.Text('Amateur Telescope Making (ATM) workshop')],
        [sg.Text('Washington D.C., U.S.A.')],
        [sg.Text("Suggestions/Feedback: ", size=(25, 1), justification="left", font=('Verdana', 10, 'bold'), key="-AUTHOR-")],
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
            if CAM_DEBUG:
               print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                if CAM_DEBUG:
                   print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                if CAM_DEBUG:
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
    global step_counter
    global prev_step_counter
    global use_circular_hough_transform
    global autosave

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

                preprocess_frame = None
                # color or grayscale?
                if not color_video:
                        preprocess_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                        preprocess_frame = frame

                # color or grayscale?
                if raw_video:
                        fkesa_frame = preprocess_frame
                else:
                   #if counter % 5 == 0:
                   #     if counter > 100:
                   #        counter = 0
                        if process_fkesa:
                           step_user_text = f"Step {step_counter}"
                        else:
                           step_user_text = ""
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
                        builder.with_param('-STEP SIZE-', step_size_val)
                        builder.with_param('user_text', step_user_text)
                        builder.with_param('debug', is_debugging)
                        builder.with_param('adaptive_find_mirror', False)
                        builder.with_param('enable_disk_rwx_operations', autosave)
                        if start_point and end_point:
                            builder.with_param('start_point', start_point)
                            builder.with_param('end_point', end_point)
                        else:
                            print("start_point and end_point None")
                            builder.with_param('start_point', (0,0))
                            builder.with_param('end_point', (640,480))
                        # ... Include other parameters as needed

                        # Build and execute the operation
                        if use_circular_hough_transform:
                            _,fkesa_frame = builder.build_auto(frame)
                        else:
                            _,fkesa_frame = builder.build_manual(frame)
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
                   #else:
                   #     fkesa_frame = preprocess_frame

                if fkesa_frame is None:
                   continue

                global shared_frame
                if fkesa_frame is not None:
                   if is_auto:
                      step_counter += 1
                      is_auto=False
                   shared_frame = fkesa_frame.copy()
                   #Record if flag set
                   if is_recording:
                      if out is not None:
                         #print("Continuing Video Recording....")
                         if raw_video:
                            shared_frame = cv2.resize(shared_frame, (640, 480))
                         else:
                            #Video reording writer needs RGB or it silently fails (flaw in OpenCV ver 2)
                            shared_frame = cv2.cvtColor(shared_frame, cv2.COLOR_BGRA2RGB)
                         out.write(shared_frame)
                   #elif not is_recording:
                   #   if out is not None:
                   #      print("Stopping Video Recording....")
                   #      out.release()
                   #      out = None
        except Exception as e:
               print(f"An exception occurred {e}")
      

    cap.release()
    cv2.destroyAllWindows()


def generate_please_wait_image():
    # Create a blank white image
    image = np.ones((640, 480, 3), dtype=np.uint8) * 255

    # Add "Please Wait" text
    text = "Please Wait..."
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

    return image

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
distance_inches = 0.10  # Replace with the distance in inches you want to convert

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

#================ Enable/Disable circular Hough Transform ==================
def enable_all_CHT_widgets(window):
    widgets_to_enable = ['-MINDIST SLIDER-', '-PARAM SLIDER A-', '-PARAM SLIDER B-', '-RADIUS SLIDER A-', '-RADIUS SLIDER B-']
    for widget_key in widgets_to_enable:
        window[widget_key].update(disabled=False)

def disable_all_CHT_widgets(window):
    widgets_to_disable = ['-MINDIST SLIDER-', '-PARAM SLIDER A-', '-PARAM SLIDER B-', '-RADIUS SLIDER A-', '-RADIUS SLIDER B-']
    for widget_key in widgets_to_disable:
        window[widget_key].update(disabled=True)
 
#================= Stepper motor distance conversion ================
#Function to re-enable or disable measurement widgets

def disable_all_measurements_widgets(window):
    window['-MEASUREMENTS-'].update(button_color=('orange', 'black'), text='Stop Measurements')
    widgets_to_disable = ['-DIA TEXT-', '-FL TEXT-', '-AUTOFOUCAULT-', '-STEP SIZE-', '-STEP DELAY-', '-MAX ATTEMPTS-']
    for widget_key in widgets_to_disable:
        window[widget_key].update(disabled=True)
        
def enable_all_measurements_widgets(window):
    window['-MEASUREMENTS-'].update(button_color=('black', 'orange'), text='Start Measurements')
    widgets_to_enable = ['-DIA TEXT-', '-FL TEXT-', '-AUTOFOUCAULT-', '-STEP SIZE-', '-STEP DELAY-', '-MAX ATTEMPTS-']
    for widget_key in widgets_to_enable:
        window[widget_key].update(disabled=False)

#================= Auto-Foucault process ================

# Function to disable all widgets in the window
def disable_all_autofoucault_widgets(window):
    widgets_to_disable = ['-DIA TEXT-', '-FL TEXT-', '-AUTOFOUCAULT-', '-MEASUREMENTS-', '-MEASUREMENTS CSV-', '-STEP SIZE-', '-STEP DELAY-', '-MAX ATTEMPTS-',
                          '-PAUSE PLAY VIDEO-', '-RECORD VIDEO-', '-RAW VIDEO SELECT-', '-COLOR VIDEO SELECT-', '-CAMERA SELECT-']
    for widget_key in widgets_to_disable:
        window[widget_key].update(disabled=True)

# Function to re-enable all widgets in the window
def enable_all_autofoucault_widgets(window):
    widgets_to_enable = ['-DIA TEXT-', '-FL TEXT-', '-AUTOFOUCAULT-', '-MEASUREMENTS-', '-MEASUREMENTS CSV-', '-STEP SIZE-', '-STEP DELAY-', '-MAX ATTEMPTS-',
                         '-PAUSE PLAY VIDEO-', '-RECORD VIDEO-', '-RAW VIDEO SELECT-', '-COLOR VIDEO SELECT-', '-CAMERA SELECT-']
    for widget_key in widgets_to_enable:
        window[widget_key].update(disabled=False)

autofoucault_error_lut = {
    0: "No error",
    1: "Cannot set CW steps (X)",
    2: "Cannot set CW delay (X)",
    3: "Cannot find start position - limit switch (X) ",
    4: "Max attempts to find start position - limit switch reached (X)",
    5: "Cannot reset counters",
    6: "Cannot find end position - limit switch (X) ",
    7: "Max attempts to find end position - limit switch reached (X)",
    8: "Autofoucault failed!"
}

def autofoucault_set_steps(helper, device_ip="192.168.4.1", result_steps=500):
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
        return True, 0

def autofoucault_set_delay(helper, device_ip="192.168.4.1", result_delay_usec=50):
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
        return True, 0

def autofoucault_goto_limit_begin_x(helper, device_ip="192.168.4.1", max_attempts=50):
        found_start_x = False
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
        return True, 0

def autofoucault_reset_counters(helper, device_ip="192.168.4.1"):
        # Reset Counters
        url_string = f"http://{device_ip}/reset"
        result_string = helper.get_string_value_from_url(url_string)
        if result_string is not None:
           print("Received string value:", result_string)
        else:
           print("no response!")
           return False, 5
        return True, 0


def autofoucault_goto_limit_end_x(helper, device_ip="192.168.4.1", max_attempts=50):
        global is_auto
        global step_counter
        global prev_step_counter
        with lock:
             step_counter=0
             prev_step_counter=0
        found_end_x = False
        for num in range(0, max_attempts):
                url_boolean = f"http://{device_ip}/reached_end_x"
                # Call the boolean method on the instance
                result_boolean = helper.get_boolean_value_from_url(url_boolean)
                if result_boolean is not None:
                    #print("Received boolean value:", result_boolean)
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
                            # ====== FKESA v2 process iteration begin =========
                            with lock:
                               is_auto = True
                            while True:
                                  if step_counter > prev_step_counter:
                                     prev_step_counter = step_counter
                                     break
                            # ====== FKESA v2 process iteration end =========
                        else:
                            print("no response!")
                            return False, 7
                #Allow some time for carriage to move along the stepper motor rail
                time.sleep(1)
                #is_auto = False
        return True, 0

def autofoucault_start(helper, device_ip="192.168.4.1", max_attempts=50):
        global is_auto
        global step_counter
        global prev_step_counter
        with lock:
             step_counter=0
             prev_step_counter=0
        found_end_x = False
        for num in range(0, max_attempts):
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
                        # ====== FKESA v2 process iteration begin =========
                        with lock:
                             is_auto = True
                        while True:
                              if step_counter > prev_step_counter:
                                 prev_step_counter = step_counter
                                 break
                        # ====== FKESA v2 process iteration end =========
                    else:
                        print("no response!")
                        return False, 8
                    #Allow some time for carriage to move along the stepper motor rail
                    time.sleep(1)
                    #is_auto = False
        return True, 0

   

def process_fkesa_v2(device_ip="192.168.4.1", result_delay_usec=50, result_steps=500, max_attempts=50):
        global exit_event
        global thread
        global auto_thread
        global auto_return
        global auto_error
        global is_auto
        global process_fkesa
        #with lock:
        #         is_auto=True
        #Default IP 192.168.4.1
        helper = FKESAHelper()
        #result_steps = inches_to_steps(distance_inches, steps_per_revolution, microsteps)
        try:
                with lock:
                     process_fkesa = True
                # Steps
                ret,_ = autofoucault_set_steps(helper,device_ip, result_steps)
                if not ret:
                   print("no response!")
                   with lock:
                            is_auto=False
                            auto_return = False
                            auto_error = 1
                   #return False, 1
                # Microsecs
                ret,_ = autofoucault_set_delay(helper,device_ip, result_delay_usec)
                if not ret:
                   print("no response!")
                   with lock:
                            is_auto=False
                            auto_return = False
                            auto_error = 2
                   #return False, 2
                #Commenting since Guy mentioned he doesn't want limit switches
                """
                ret,_=autofoucault_goto_limit_begin_x(helper,device_ip, max_attempts)
                if not ret:
                   #raise ValueError('ERROR: Could not find start reference!')
                   print('ERROR: Max attempts reached! Could not find start reference [Auto-Foucault]!')
                   with lock:
                            is_auto=False
                            auto_return = False
                            auto_error = 3
                   #return auto, 3
                #Commenting since Guy mentioned he doesn't want limit switches
                """
                # Reset Counters
                # Microsecs
                ret,_ = autofoucault_reset_counters(helper,device_ip)
                if not ret:
                   print("no response!")
                   with lock:
                            is_auto=False
                            auto_return = False
                            auto_error = 5
                   #return False, 5
                # FKESA v2 process begins below now we have carriage at the start of the rails.
                # Assuming limit switches hardware
                #ret,_=autofoucault_goto_limit_end_x(helper,device_ip, max_attempts)
                # Assuming no limit switches hardware
                ret,_=autofoucault_start(helper,device_ip, max_attempts)
                if not ret:
                   #raise ValueError('ERROR: Could not find end reference!')
                   print('ERROR: Max attempts reached! Could not find end reference [Auto-Foucault]!')
                   with lock:
                            is_auto=False
                            auto_return = False
                            auto_error = 8
                   #return False, 8
        except Exception as e:
               print(str(e))
        with lock:
           is_auto=False
           auto_return = False
           auto_error = 0
        #return True, 0

#process_fkesa_v2("192.168.4.1",50,100,10)

def autofoucault_no_uc(helper, device_ip="192.168.4.1", max_attempts=50):
        global is_auto
        global step_counter
        global prev_step_counter
        with lock:
             step_counter=0
             prev_step_counter=0
        found_end_x = False
        for num in range(0, max_attempts):
                    # ====== FKESA v2 process iteration begin =========
                    with lock:
                         is_auto = True
                    while True:
                          if step_counter > prev_step_counter:
                             prev_step_counter = step_counter
                             break
                    # ====== FKESA v2 process iteration end =========
                    #Allow some time for carriage to move along the stepper motor rail
                    time.sleep(1)
                    #is_auto = False
        return True, 0

def process_fkesa_v2_quick_test(device_ip="192.168.4.1", result_delay_usec=50, result_steps=500, max_attempts=50):
        global exit_event
        global thread
        global auto_thread
        global auto_return
        global auto_error
        global is_auto
        global process_fkesa
        with lock:
                 is_auto=True
        #Default IP 192.168.4.1
        helper = FKESAHelper()
        #result_steps = inches_to_steps(distance_inches, steps_per_revolution, microsteps)
        try:
                with lock:
                     process_fkesa = True
                # Assuming no limit switches hardware
                ret,_=autofoucault_no_uc(helper,device_ip, max_attempts)
                if not ret:
                   #raise ValueError('ERROR: Could not find end reference!')
                   print('ERROR: Max attempts reached! Could not find end reference [Auto-Foucault]!')
                   with lock:
                            is_auto=False
                            auto_return = False
                            auto_error = 8
                   #return False, 8
        except Exception as e:
               print(str(e))
        with lock:
            is_auto=False
            auto_return = False
            auto_error = 0
        #return True, 0

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
        [sg.Text("Select Images Folder",font=('Verdana', 10, 'bold'), text_color="darkred"),], 
        [
            #sg.Text("Select Images Folder",font=('Verdana', 12, 'bold'), text_color="darkred"), 
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
        [sg.Text("Click To View An Image From The Left Pane", font=('Verdana', 10, 'bold'), text_color="darkred")],
        [sg.Text(size=(70, 1), key="-TOUT-")],
        [sg.Image(key="-LOAD IMAGE-",size=(200,200))],
    ]


    # Define the window layout
    layout = [
            [sg.Image(filename='fkesa.ico.png'), sg.VerticalSeparator(), sg.Text("Foucault Knife-Edge Shadogram Analyzer (FKESA) Version 2", size=(61, 1), justification="left", font=('Verdana', 10, 'bold'),text_color='darkgreen'), sg.VerticalSeparator(),sg.Text("[]", key="-MESSAGE-", size=(120, 1), justification="left", font=('Verdana', 10, 'bold'),text_color='red'), sg.VerticalSeparator()],
        [sg.Menu(menu_def, background_color='lightblue',text_color='navy', disabled_text_color='yellow', font='Verdana', pad=(10,10))],
        [sg.HorizontalSeparator()],  # Separator 
        #[sg.Image(filename="", key="-IMAGE-", size=(640,480), enable_events=True), 
        [sg.Graph((640,480), (0, 0), (640,480), enable_events=True, key='-IMAGE-'),
         sg.VerticalSeparator(), 
         sg.Column(file_list_column), 
         sg.VerticalSeparator(), 
         sg.Column(image_viewer_column),],
        [
            [
             sg.Button('Start Recording', key='-RECORD VIDEO-',button_color = ('white','red')), 
             sg.VerticalSeparator(), 
             sg.Button('Pause Video', key='-PAUSE PLAY VIDEO-',button_color = ('white','green')) , 
             sg.VerticalSeparator(), 
             sg.Button("Save Image", size=(15, 1), button_color = ('white','blue')), 
             sg.VerticalSeparator(), 
             sg.Checkbox('Auto Save', default=False, enable_events=True, key='-AUTOSAVE SELECT-',font=('Verdana', 10, 'bold')), 
             sg.VerticalSeparator(), 
             sg.Text("Step Size (Inches)", size=(15, 1), justification="left", font=('Verdana', 10, 'bold'), key="-STEP SIZE INCHES-"),
             sg.InputText('0.10', key='-STEP SIZE-', size=(8, 1), enable_events=True, justification='center', tooltip='Enter an integer or floating-point number'),
             sg.VerticalSeparator(),  # Separator 
             sg.Text("Step Delay (μsecs)", size=(16, 1), justification="left", font=('Verdana', 10, 'bold'), key="-PULSE DELAY-"),
             sg.InputText('50', key='-STEP DELAY-', size=(8, 1), enable_events=True, justification='center', tooltip='Enter an integer number'),
             sg.VerticalSeparator(),  # Separator 
             sg.Text("Max Steps", size=(9, 1), justification="left", font=('Verdana', 10, 'bold'), key="-MAX STEPS-"),
             sg.InputText('10', key='-MAX ATTEMPTS-', size=(8, 1), enable_events=True, justification='center', tooltip='Enter an integer number'),
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
             sg.Button('Select Camera'), 
             sg.VerticalSeparator(), 
             sg.Checkbox('Raw Video', default=True, enable_events=True, key='-RAW VIDEO SELECT-',font=('Verdana', 10, 'bold')), 
             sg.VerticalSeparator(), 
             sg.Checkbox('Colored Raw Video', default=True, enable_events=True, key='-COLOR VIDEO SELECT-', font=('Verdana', 10, 'bold')), 
             sg.VerticalSeparator(),  # Separator 
             sg.Checkbox('Grids', default=False, enable_events=True, key='-GRIDS-',font=('Verdana', 10, 'bold')), 
             sg.VerticalSeparator(), 
             sg.Button('Clear Circle', key='-CLEAR CIRCLE-',button_color = ('black','lightgreen'),disabled=False),
             sg.VerticalSeparator(), 
             sg.Text("Diameter (Inches) [Default: 6]", size=(25, 1), justification="left", font=('Verdana', 10, 'bold'), key="-DIAMETER TEXT-"),
             sg.InputText('6.0', key='-DIA TEXT-', size=(8, 1), enable_events=True, justification='center', tooltip='Enter an integer or floating-point number'),
             sg.VerticalSeparator(),  # Separator 
             sg.Text("Focal Length (Inches) [Default: 48]", size=(30, 1), justification="left", font=('Verdana', 10, 'bold'), key="-FOCAL LENGTH TEXT-"),
             sg.InputText('48.0', key='-FL TEXT-', size=(8, 1), enable_events=True, justification='center', tooltip='Enter an integer or floating-point number'),
             sg.VerticalSeparator(),  # Separator 
            ],
            #[sg.Button('SELECT CAMERA'), sg.VerticalSeparator(), sg.Button('Cancel'), sg.VerticalSeparator()], 
        ],
        [sg.HorizontalSeparator()],  # Separator 
        [
            sg.Text("Circular Hough Transform (CHT) Parameters [Mirror Detection]", size=(80, 1), justification="left", font=('Verdana', 10, 'bold'), text_color='darkred'),
            sg.Checkbox('Use CHT', default=False, enable_events=True, key='-USE CHT-',font=('Verdana', 10, 'bold')), 
        ],
        [sg.HorizontalSeparator()],  # Separator 
        [
            sg.Text("Minimum Distance (Pixels) [Default: 50]", size=(40, 1), justification="left", font=('Verdana', 10, 'bold'), key="-MINDIST A-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (1, 255),
                50,
                1,
                orientation="h",
                enable_events=True,
                size=(50, 10),
                key="-MINDIST SLIDER-",
                font=('Verdana', 10, 'normal'),
                # text_color=('darkgreen') # experimental
            ),
            sg.VerticalSeparator(),  # Separator 
            sg.Text("Processing Delay Milliseconds [Default: 200]", size=(40, 1), justification="left", font=('Verdana', 10, 'bold'), key="-MINDIST B-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (0,1000),
                200,
                100,
                orientation="h",
                enable_events=True,
                size=(50, 10),
                key="-DELAY SLIDER-",
                font=('Verdana', 10, 'normal'),
            ),
            sg.VerticalSeparator(),  # Separator 
        ],
        [
            sg.Text("Parameter 1 (Pixels) [Default: 25]", size=(40, 1), justification="left", font=('Verdana', 10, 'bold'), key="-PARAMS A-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (1, 255),
                25,
                1,
                orientation="h",
                enable_events=True,
                size=(50, 10),
                key="-PARAM SLIDER A-",
                font=('Verdana', 10, 'normal'),
            ),
            sg.VerticalSeparator(),  # Separator 
            sg.Text("Parameter 2 (Pixels) [Default: 60]", size=(40, 1), justification="left", font=('Verdana', 10, 'bold'), key="-PARAMS B-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (1, 255),
                60,
                1,
                orientation="h",
                enable_events=True,
                size=(50, 10),
                key="-PARAM SLIDER B-",
                font=('Verdana', 10, 'normal'),
            ),
            sg.VerticalSeparator(),  # Separator 
        ],
        [
            sg.Text("Minimum Radius (Pixels) [Default: 10]", size=(40, 1), justification="left", font=('Verdana', 10, 'bold'), key="-MIN RADIUS-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (1, 255),
                20,
                1,
                orientation="h",
                enable_events=True,
                size=(50, 10),
                key="-RADIUS SLIDER A-",
                font=('Verdana', 10, 'normal'),
            ),
            sg.VerticalSeparator(),  # Separator 
            sg.Text("Maximum Radius (Pixels) [Default: 0]", size=(40, 1), justification="left", font=('Verdana', 10, 'bold'), key="-MAX RADIUS-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (0, 255),
                0,
                1,
                orientation="h",
                enable_events=True,
                size=(50, 10),
                key="-RADIUS SLIDER B-",
                font=('Verdana', 10, 'normal'),
            ),
            sg.VerticalSeparator(),  # Separator 
        ],
        [sg.HorizontalSeparator()],  # Separator 
        [sg.Text("Intensity Parameters [Null Zones Identification]", size=(80, 1), justification="left", font=('Verdana', 10, 'bold'), text_color='darkred')],
        [sg.HorizontalSeparator()],  # Separator 
        [
            sg.Text("Brightness Tolerance [Default: 10]", size=(40, 1), justification="left", font=('Verdana', 10, 'bold'), key="-INTENSITY PARAMS A-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (0, 50),
                10,
                1,
                orientation="h",
                enable_events=True,
                size=(50, 10),
                key="-BRIGHTNESS SLIDER-",
                font=('Verdana', 10, 'normal'),
            ),
            sg.VerticalSeparator(),  # Separator 
            sg.Text("Number Of Zones [Default: 50]", size=(40, 1), justification="left", font=('Verdana', 10, 'bold'), key="-INTENSITY PARAMS B-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (30, 50),
                60,
                1,
                orientation="h",
                enable_events=True,
                size=(50, 10),
                key="-ZONES SLIDER-",
                font=('Verdana', 10, 'normal'),
            ),
            sg.VerticalSeparator(),  # Separator 
        ],
        [
            sg.Text("Angel (Slice Or Pie) (Degrees) [Default: 10]", size=(40, 1), justification="left", font=('Verdana', 10, 'bold'), key="-ANGLE-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (10, 90),
                10,
                1,
                orientation="h",
                size=(50, 10),
                enable_events=True,
                key="-ANGLE SLIDER-",
                font=('Verdana', 10, 'normal'),
            ),
            sg.VerticalSeparator(),  # Separator 
            sg.Text("Skip Zones From The Center [Default: 10]", size=(40, 1), justification="left", font=('Verdana', 10, 'bold'), key="-SKIP ZONES-"),
            sg.VerticalSeparator(),  # Separator 
            sg.Slider(
                (1, 20),
                10,
                1,
                orientation="h",
                size=(50, 10),
                enable_events=True,
                key="-SKIP ZONES SLIDER-",
                font=('Verdana', 10, 'normal'),
            ),
            sg.VerticalSeparator(),  # Separator 
        ],
        [sg.HorizontalSeparator()],  # Separator 
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
        elif event == "-IMAGE-":
          #mouse_x, mouse_y = window.CurrentLocation()
          #print(f"Clicked inside the Window at ({mouse_x}, {mouse_y})")
          x, y = values['-IMAGE-']
          with lock:
            """
            if start_point is None:
                start_point = (x, y)
            else:
                end_point = (x, y)
                #draw_square(window['-IMAGE-'], start_point, end_point)
            print(start_point)
            print(end_point)
            """
            if len(points) < 3:
               points.append((x, y))
            #if len(points) == 3:
            #   draw_circle_thru_3_pts(window['-IMAGE-'], points)
        elif event == '-CLEAR CIRCLE-':
          with lock:
            start_point = None
            end_point = None
            points = []
            window['-IMAGE-'].erase()
            print('clear circle')
        elif event == "-AUTOFOUCAULT-":
           window['-MESSAGE-'].update('[AUTOFOUCAULT COMMENCED...PLEASE WAIT A FEW SECONDS/MINUTES FOR THE OPERATION TO COMPLETE...]')
           confirm_proceed = sg.popup_yes_no("Is your Foucault setup ready? Are you sure you want to proceed with AutoFoucault?\n\nNote that this test may take a few minutes to complete. You will not be able to use other widgets on this GUI during this operation.")
           if confirm_proceed == "Yes" and \
              is_valid_number(values['-STEP SIZE-']) and \
              is_valid_integer(values['-STEP DELAY-']) and \
              is_valid_integer(values['-MAX ATTEMPTS-']) and \
              is_valid_mirror_params(values['-DIA TEXT-']) and \
              is_valid_mirror_params(values['-FL TEXT-']):
              #proceed

              #window['-MESSAGE-'].update('[Autofoucault ongoing....Please wait...]')
              with lock:
                      #window['-MESSAGE-'].update('[Autofoucault ongoing....Please wait...]')
                      # First stop any ongoing measurements
                      if is_measuring:
                        print("Stopping measurements.....") 
                        enable_all_measurements_widgets(window)
                        is_measuring = False
              # Disable all Widgets temporarily
              disable_all_autofoucault_widgets(window)
              distance_inches=float(values['-STEP SIZE-'])
              #result_steps = inches_to_steps(distance_inches, stepper_steps_per_revolution, stepper_microsteps)
              #ball_screw_pitch_mm = 5
              distance_mm = inches_to_mm(distance_inches)
              result_steps = distance_to_steps(distance_mm, stepper_steps_per_revolution, stepper_microsteps, ball_screw_pitch_mm)
              #TODO - add exception
              result_delay_usec = int(values['-STEP DELAY-'])
              result_max_attempts = int(values['-MAX ATTEMPTS-'])
              #Add 1 to max attempts
              result_max_attempts += 1
              print(result_max_attempts)
              #success, error = process_fkesa_v2(device_ip="192.168.4.1", result_delay_usec=result_delay_usec, result_steps=result_steps, max_attempts=50)
              #success, error = process_fkesa_v2_test(device_ip="192.168.4.1", result_delay_usec=result_delay_usec, result_steps=result_steps, max_attempts=5)
              #print(success,error)
              # Start the thread function when the "Start Thread" button is pressed

              if not autofoucault_simple_simulation:
                 auto_thread = threading.Thread(target=process_fkesa_v2, args=("192.168.4.1",), kwargs={"result_delay_usec": result_delay_usec, "result_steps": result_steps, "max_attempts": result_max_attempts})
              else:   
                 auto_thread = threading.Thread(target=process_fkesa_v2_quick_test, args=("192.168.4.1",), kwargs={"result_delay_usec": result_delay_usec, "result_steps": result_steps, "max_attempts": result_max_attempts})
              #auto_thread = threading.Thread(target=process_fkesa_v2_quick_test, args=("192.168.4.1",), kwargs={"result_delay_usec": result_delay_usec, "result_steps": result_steps, "max_attempts": 5})
              auto_thread.daemon = True
              auto_thread.start()
              """
              if not success:
                    sg.popup_ok(f"FKESA AUTOFOUCAULT Failed with an error # {error} -> \"{autofoucault_error_lut[error]}\". Click OK to continue.") 
                    window['-MESSAGE-'].update('[*AN ERROR OCCURRED*: AUTOFOUCAULT FAILED!!!]')
              with lock:
                      # Re-enable all Widgets
                      enable_all_autofoucault_widgets(window)
                      window['-MESSAGE-'].update('[]')
                      sg.popup_ok(f"FKESA AUTOFOUCAULT process Finished. Click OK to continue.") 
              """
           else:
               window['-MESSAGE-'].update('[]')
               sg.popup_ok(f"FKESA AUTOFOUCAULT process cannot be started. Please check and enter valid values for step size, step delay, max steps, focal length and diameter. Click OK to continue.") 
           #sys.exit()
        elif event == "-MEASUREMENTS-":
            with lock:
             if not is_measuring:
                if check_step_size_validity(values):
                   print("Starting measurements.....") 
                   step_size_val = float(values['-STEP SIZE-'])
                   step_delay_val = float(values['-STEP DELAY-'])
                   max_attempts_val = int(values['-MAX ATTEMPTS-'])
                   disable_all_measurements_widgets(window)
                   measurement_run_counter+=1
                   is_measuring = True
                else:
                   sg.popup_error('Invalid input! Please enter a valid integer or floating-point number.')
             elif is_measuring:
                print("Stopping measurements.....") 
                enable_all_measurements_widgets(window)
                is_measuring = False
        elif event == "-MEASUREMENTS CSV-":
             if not is_another_file_instance_running('measurement_csv'):
                try:
                    create_lock_file('measurement_csv')  # Create lock file
                    window['-MEASUREMENTS CSV-'].update(disabled=True, text='Viewing Measurements Data')
                    python_executable = sys.executable
                    subprocess.run([python_executable, './FKESA_v2_csv.py'])
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
                    python_executable = sys.executable
                    subprocess.run([python_executable, './FKESA_v2_optics.py', '--radius', str(diameter_val/2), '--focal_length', str(focal_length_val)])
                except Exception as e:
                    sg.popup_error(f"Error running /FKESA_v2_optics.py: {e}")
                finally:
                    remove_lock_file('optics')  # Remove lock file
                    window['-OPTICS-'].update(disabled=False, text='Optical Ray Diagram')
        elif event == "-AUTOSAVE SELECT-":
             if not values["-AUTOSAVE SELECT-"]:
                autosave = False
             elif values["-AUTOSAVE SELECT-"]:
                autosave = True
        elif event == "-RECORD VIDEO-":
             if not is_recording:
                print("Starting video recording.....") 
                window['-RECORD VIDEO-'].update(button_color = ('black','yellow'))
                window['-RECORD VIDEO-'].update(text = ('Stop Recording'))
                video_filename = f"fkesa_v2_{int(time.time())}.avi"  # Generate a filename (you can adjust this)
                if out is not None:
                   out.release()
                   out = None
                if raw_video:
                   out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640,  480))
                else:
                   #out = cv2.VideoWriter(video_filename, fourcc, 1.0, (640,  480))
                   fps = calculate_fps(fkesa_time_delay)
                   out = cv2.VideoWriter(video_filename, fourcc, float(fps), (640,  480))
                is_recording = True
             elif is_recording:
                print("Stopping video recording.....") 
                window['-RECORD VIDEO-'].update(button_color = ('white','red'))
                window['-RECORD VIDEO-'].update(text = ('Start Recording'))
                if out is not None:
                   out.release()
                   out = None
                is_recording = False
        elif event == "-PAUSE PLAY VIDEO-":
             if is_playing:
                window['-PAUSE PLAY VIDEO-'].update(button_color = ('black','yellow'))
                window['-PAUSE PLAY VIDEO-'].update(text = ('Play Video'))
                is_playing = False
                print("Pausing the worker thread...")
                exit_event.set()  # Set the exit event to stop the loop
                # Wait for the processing thread to complete before closing the window
                if thread is not None and thread.is_alive():
                   thread.join()
             elif not is_playing:
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
        elif event == "-GRIDS-":
             if not values["-GRIDS-"]:
                grids = False
             elif values["-GRIDS-"]:
                grids = True
        elif event == "-RAW VIDEO SELECT-":
             if not values["-RAW VIDEO SELECT-"]:
                window['-MEASUREMENTS-'].update(disabled=False)
                window['-AUTOFOUCAULT-'].update(disabled=False)
                raw_video = False
             elif values["-RAW VIDEO SELECT-"]:
                window['-MEASUREMENTS-'].update(disabled=True)
                window['-AUTOFOUCAULT-'].update(disabled=True)
                raw_video = True
                if is_measuring:
                  enable_all_measurements_widgets(window)
                  is_measuring = False
        elif event == "-COLOR VIDEO SELECT-":
             if not values["-COLOR VIDEO SELECT-"]:
                color_video = False
             elif values["-COLOR VIDEO SELECT-"]:
                color_video = True
        elif event == "-USE CHT-":
             if not values["-USE CHT-"]:
                print('CHT dis')
                disable_all_CHT_widgets(window)
                use_circular_hough_transform = False
             elif values["-USE CHT-"]:
                print('CHT en')
                enable_all_CHT_widgets(window)
                use_circular_hough_transform = True
        elif event == "-DELAY SLIDER-":
             fkesa_time_delay = int(values["-DELAY SLIDER-"])
        # Inside the main event loop where the sliders are handled
        elif event == "-MINDIST SLIDER-" \
             or event == "-PARAM SLIDER A-" or event == "-PARAM SLIDER B-" \
             or event == "-RADIUS SLIDER A-" or event == "-RADIUS SLIDER B-" \
             or event == "-BRIGHTNESS SLIDER-" or event == "-ZONES SLIDER-" \
             or event == "-ANGLE SLIDER-" or event == "-SKIP ZONES SLIDER-" \
             or event == "-DIA TEXT-" or event == "-FL TEXT-" :
            with lock:
              mindist_val = int(values["-MINDIST SLIDER-"])
              param_a_val = int(values["-PARAM SLIDER A-"])
              param_b_val = int(values["-PARAM SLIDER B-"])
              radius_a_val = int(values["-RADIUS SLIDER A-"])
              radius_b_val = int(values["-RADIUS SLIDER B-"])
              brightness_tolerance_val = int(values["-BRIGHTNESS SLIDER-"])
              zones_val = int(values["-ZONES SLIDER-"])
              angle_val = int(values["-ANGLE SLIDER-"])
              #diameter_val = float(values["-DIA TEXT-"])
              #focal_length_val = float(values["-FL TEXT-"])
              diameter_val = values.get("-DIA TEXT-", "1.0")  # Default diameter value is 1.0 if input is empty
              focal_length_val = values.get("-FL TEXT-", "1.0")  # Default focal length value is 1.0 if input is empty
              try:
                 diameter_val = float(diameter_val)
                 focal_length_val = float(focal_length_val)
                 if diameter_val < 1.0:
                    diameter_val = 1.0
                 if focal_length_val < 1.0:
                    focal_length_val = 1.0
              except ValueError:
                 sg.popup_error("Invalid input! Please enter valid floating-point numbers.")
       
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
                   save_directory = os.path.join(home_dir, 'FKESAv2Images')
                   os.makedirs(save_directory, exist_ok=True)
                   image_path = os.path.join(save_directory, filename)
                   #image_path = os.path.join(home_dir, 'Desktop', filename)
                   print(image_path)
                   #sys.exit(1)
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
            if is_measuring:
                    sg.popup_ok("Please stop all measurements before switching cameras. Click OK to continue.") 
            elif not is_measuring:
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
                    if thread is not None and thread.is_alive():
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
        #if is_auto:
        #    # Create a green image
        #    green_image = np.zeros((480, 640, 3), dtype=np.uint8)
        #    green_image[:, :, 1] = 255  # Set the green channel to maximum intensity
        #    # Update the GUI window with the green image
        #    imgbytes = cv2.imencode('.png', green_image)[1].tobytes()
        #    window['-IMAGE-'].update(data=imgbytes)

        # Update the GUI from the main thread
        with lock:
          if 'shared_frame' in globals():
            if shared_frame is not None and window is not None:
               imgbytes = cv2.imencode('.png', shared_frame)[1].tobytes()
               #window['-IMAGE-'].update(data=imgbytes)
               window['-IMAGE-'].erase()
               window['-IMAGE-'].draw_image(data=imgbytes, location=(0, 480))
               # Help the user specify the grids for the mirror
               if grids:
                  draw_mesh_grid(window['-IMAGE-'],40,40,color='white')
               # Let user specify the ROI for the mirror
               #if start_point and end_point:
               #   draw_square(window['-IMAGE-'], start_point, end_point)
               if len(points) == 3:
                  draw_circle_thru_3_pts(window['-IMAGE-'], points)

            #else:
            #   # Generate "please wait" image
            #   please_wait_image = generate_please_wait_image()
            #   imgbytes = cv2.imencode('.png', please_wait_image)[1].tobytes()
            #   window['-IMAGE-'].update(data=imgbytes)
        # Update the input background color based on validity
        input_background_color = 'white' if is_valid_number(values['-STEP SIZE-']) else 'pink'
        window['-STEP SIZE-'].update(background_color=input_background_color)
        input_background_color = 'white' if is_valid_integer(values['-STEP DELAY-']) else 'pink'
        window['-STEP DELAY-'].update(background_color=input_background_color)
        input_background_color = 'white' if is_valid_integer(values['-MAX ATTEMPTS-']) else 'pink'
        window['-MAX ATTEMPTS-'].update(background_color=input_background_color)
        input_background_color = 'white' if is_valid_mirror_params(values['-DIA TEXT-']) else 'pink'
        window['-DIA TEXT-'].update(background_color=input_background_color)
        input_background_color = 'white' if is_valid_mirror_params(values['-FL TEXT-']) else 'pink'
        window['-FL TEXT-'].update(background_color=input_background_color)

        #Error
        with lock:
           if process_fkesa and auto_error >= 0:
                if auto_error > 0:
                   sg.popup_ok(f"FKESA AUTOFOUCAULT Failed with an error # {auto_error} -> \"{autofoucault_error_lut[auto_error]}\". Click OK to continue.") 
                   window['-MESSAGE-'].update('[*AN ERROR OCCURRED*: AUTOFOUCAULT FAILED!!!]')
                process_fkesa = False
                auto_error = -1
                # Re-enable all Widgets
                enable_all_autofoucault_widgets(window)
                window['-MESSAGE-'].update('[]')
                sg.popup_ok(f"FKESA AUTOFOUCAULT process Finished. Click OK to continue.") 


    # Wait for the autofoucault processing thread to complete before closing the window
    if auto_thread is not None and auto_thread.is_alive():
       auto_thread.join()

    # Wait for the processing thread to complete before closing the window
    if thread is not None and thread.is_alive():
       thread.join()

    if cap is not None:
       cap.release()  # Release the camera explicitly

    if out is not None:
       out.release()

    window.close()

except Exception as e:
        print(f"An error occurred: {e}")

