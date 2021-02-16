import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from expressionRecognition import expression_recognition

window_height = 130
window_width = 550
error_window_height = 130
error_window_width = 550
button_width = 15

font = ('arial', 11)
msg_txt = "You can either analyse the facial expressions in a video or use the webcam."
video_error_txt = "The file was not readable. Please select a video file (e.g. mp4)."
webcam_error_txt = "It was not possible to access the webcam. Please connect a working webcam."
video_button_txt = "Select Video"
webcam_button_txt = "Use Webcam"
error_button_txt = "Ok"
gui_title = "Facial expression recognition"
error_title = "Error"

#create window
gui = tk.Tk()
gui.title(gui_title)
gui.configure(bg='white')
screen_width = gui.winfo_screenwidth()
screen_height = gui.winfo_screenheight()
x = int((screen_width/2) - (window_width/2))
y = int((screen_height/2) - (window_height/2))
gui.geometry("{}x{}+{}+{}".format(window_width, window_height, x, y))

def analyse_video():
    #open file dialoge so that the user can easily select a video file
    filename = askopenfilename() 
    error = expression_recognition(filename)
    if error == -1:
        error_msg(video_error_txt)

def analyse_webcam():
    #use 0 to select the video input of the main webcam
    error = expression_recognition(0)
    if error == -1:
        error_msg(webcam_error_txt)

def error_msg(error_txt):
    #create error window
    error_win = tk.Toplevel(gui)
    error_win.title(error_title)
    error_win.configure(bg='white')
    error_x = int((screen_width/2) - (error_window_width/2))
    error_y = int((screen_height/2) - (error_window_height/2))
    error_win.geometry("{}x{}+{}+{}".format(error_window_width, error_window_height, error_x, error_y))

    #display error message
    error_lable = ttk.Label(error_win, text=error_txt)
    error_lable.configure(background='white', font=font)
    error_lable.pack(pady=(30,0))

    #button to close error window
    error_button = ttk.Button(error_win, text=error_button_txt, width=button_width, command=error_win.destroy)
    error_button.pack(side='bottom', pady=(0,30))

label = ttk.Label(gui, text=msg_txt)
label.configure(background='white', font=font)
label.pack(pady=(15,0))

#button to analyse a video
video_button = ttk.Button(gui, text=video_button_txt, width=button_width, command=analyse_video)
video_button.pack(pady=(20,0))

#button to analyse the webcam input
webcam_button = ttk.Button(gui, text=webcam_button_txt, width=button_width, command=analyse_webcam)
webcam_button.pack(side='bottom', pady=(0,15))

tk.mainloop()
