import sys
import threading
import time
import tkinter as tk
import PIL
from PIL import Image, ImageTk
import cv2
import PySimpleGUI as sg
import os
from PIL import ImageFont
from PIL import ImageDraw
from io import BytesIO
import pathlib
# import YOLO model
from ultralytics import YOLO

# Load a model
model = YOLO("/home/rvp/Work_Student2024/TrainAi/runs/classify/train14/weights/best.pt") #xxx

class_dict = {0: 'Angry', 1: 'Bored', 2: 'Confused', 3: 'cool', 4: 'Errrr', 5: 'Funny', 6: 'Happy', 7: 'Normal', 8: 'Proud', 9: 'Sad', 10: 'Scared', 11: 'Shy', 12: 'Sigh', 13: 'super_angry', 14: 'Surprised', 15: 'Suspicious', 16: 'sweet', 17: 'tricky', 18: 'unhappy', 19: 'Worried'}

class_predict = []






class App:
    """
    TODO: change slider resolution based on vid length
    TODO: make top menu actually do something :P    """
    #################
    # Video methods #
    #################
    def load_video(self):
        """Start video display in a new thread"""
        thread = threading.Thread(target=self.update, args=())
        thread.daemon = 1
        thread.start()

    def update(self):
        """Update the canvas element with the next video frame recursively"""
        start_time = time.time()
        if self.vid:
            if self.play:

                # Get a frame from the video source only if the video is supposed to play
                ret, frame = self.vid.get_frame()
                results = model.predict(frame)
                print("=====>",class_dict[results[0].probs.data.argmax().item()])
                if ret:
                    self.photo = PIL.ImageTk.PhotoImage(
                        image=PIL.Image.fromarray(frame).resize((self.vid_width, self.vid_height), Image.NEAREST)
                    )
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

                    self.frame += 1
                    self.update_counter(self.frame)

            # Uncomment these to be able to manually count fps
            # print(str(self.next) + " It's " + str(time.ctime()))
            # self.next = int(self.next) + 1
        # The tkinter .after method lets us recurse after a delay without reaching recursion limit. We need to wait
        # between each frame to achieve proper fps, but also count the time it took to generate the previous frame.
        self.canvas.after(abs(int((self.delay - (time.time() - start_time)) * 1000)), self.update)

    def set_frame(self, frame_no):
        """Jump to a specific frame"""
        if self.vid:
            # Get a frame from the video source only if the video is supposed to play
            ret, frame = self.vid.goto_frame(frame_no)
            self.frame = frame_no
            self.update_counter(self.frame)

            if ret:
                self.photo = PIL.ImageTk.PhotoImage(
                    image=PIL.Image.fromarray(frame).resize((self.vid_width, self.vid_height), Image.NEAREST))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def update_counter(self, frame):
        """Helper function for updating slider and frame counter elements"""
        self.window.Element("slider").Update(value=frame)
        self.window.Element("counter").Update("{}/{}".format(frame, self.frames))    
    
    def __init__(self):

        # ------ App states ------ #
        self.play = True  # Is the video currently playing?
        self.delay = 0.023  # Delay between frames - not sure what it should be, not accurate playback
        self.frame = 1  # Current frame
        self.frames = None  # Number of frames
        # ------ Other vars ------ #
        self.vid = None
        self.photo = None
        self.next = "1"
        self.lx = 0
        self.ly = 0        

        
        folder = r"/home/rvp/Work_Student2024/video/class/Classes"  # get list of PNG files in folder
        png_files = [folder + '/' + f for f in os.listdir(folder) if '.png' in f]
        png_Addfiles = folder + '//////' + '19.png'
        image_size = (40, 40)     # image size
        headers, size = 1, 50     # table cells
        cols, rows = 1, 10        # view cells
        x0, y0 = 0, 0             # Top-left conner

        path = [[(y, x) for x in range(headers)] for y in range(size)]  # refer to image
        print("path:", path)

        def AddImg():
            if self.ly in range(rows):
                self.window[(self.ly, self.lx)].update(filename=png_Addfiles, visible=True)
                self.ly = self.ly +1
            else :
                self.window.refresh()
                for self.ly in range(rows):
                    self.window[(self.ly, self.lx)].update(filename='', visible=False)
                self.ly = 0
            return None
                
        # def AddImg():
            # self.window[(self.ly, self.lx)].update(filename= png_Addfiles)
            # self.ly = self.ly +1
                         
        def scroll():
            for y in range(rows):
                for x in range(cols):
                    if loaded:
                        self.window[(y, x)].update(filename= png_files[(y+y0) + (x+x0)])
        
        # def vscroll(event):
            # global y0
            # delta = int(event.delta/120)
            # y0 = min(max(0, y0-delta), size-rows)
            # scroll()
            # self.window['V_Scrollbar'].update(value=size-rows-y0)
            # self.window.refresh()

        def create_image(y, x):
            font = ImageFont.truetype("courbd.ttf", 30)
            im = Image.new('RGBA', image_size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(im, 'RGBA')
            draw.text((image_size[0]//2, image_size[1]//2), f'Image ({y}, {x})',
                font=font, anchor='mm')
            with BytesIO() as output:
                im.save(output, format="PNG")
                im_data = output.getvalue()
            return im_data


        table = [[sg.Image(size=(40, 40), background_color='green', key=(y, x))
            for x in range(cols)] for y in range(rows)]
        option = {'resolution':1, 'pad':(0, 0), 'disable_number_display':True,
            'enable_events':True}
            
        # ------ Menu Definition ------ #
        menu_def = [['&File', ['&Open', '&Save', '---', 'Properties', 'E&xit']],
                    ['&Edit', ['Paste', ['Special', 'Normal', ], 'Undo'], ],
                    ['&Help', '&About...']]
                    
        # รูปของ 20 class 
        file_list_column = [
            [sg.Text("Classes")],
            [sg.Image(r"/home/rvp/Work_Student2024/video/class/Image/0.png"), sg.Text("0 Angry   "),sg.Image(r"/home/rvp/Work_Student2024/video/class/Image/10.png"), sg.Text("10 Scared")],
            [sg.Image(r"/home/rvp/Work_Student2024/video/class/Image/1.png"), sg.Text("1 Bored   "),sg.Image(r"/home/rvp/Work_Student2024/video/class/Image/11.png"), sg.Text("11 Shy")],
            [sg.Image(r"/home/rvp/Work_Student2024/video/class/Image/2.png"), sg.Text("2 Confused"),sg.Image(r"/home/rvp/Work_Student2024/video/class/Image/12.png"), sg.Text("12 Sigh")],
            [sg.Image(r"/home/rvp/Work_Student2024/video/class/Image/3.png"), sg.Text("3 Cool    "),sg.Image(r"/home/rvp/Work_Student2024/video/class/Image/13.png"), sg.Text("13 Super_angry")],
            [sg.Image(r"/home/rvp/Work_Student2024/video/class/Image/4.png"), sg.Text("4 Errrr   "),sg.Image(r"/home/rvp/Work_Student2024/video/class/Image/14.png"), sg.Text("14 Surprised")],
            [sg.Image(r"/home/rvp/Work_Student2024/video/class/Image/5.png"), sg.Text("5 Funny   "),sg.Image(r"/home/rvp/Work_Student2024/video/class/Image/15.png"), sg.Text("15 Suspicious")],
            [sg.Image(r"/home/rvp/Work_Student2024/video/class/Image/6.png"), sg.Text("6 Happy   "),sg.Image(r"/home/rvp/Work_Student2024/video/class/Image/16.png"), sg.Text("16 Sweet")],
            [sg.Image(r"/home/rvp/Work_Student2024/video/class/Image/7.png"), sg.Text("7 Normal  "),sg.Image(r"/home/rvp/Work_Student2024/video/class/Image/17.png"), sg.Text("17 Tricky")],
            [sg.Image(r"/home/rvp/Work_Student2024/video/class/Image/8.png"), sg.Text("8 Proud   "),sg.Image(r"/home/rvp/Work_Student2024/video/class/Image/18.png"), sg.Text("18 Unhappy")],
            [sg.Image(r"/home/rvp/Work_Student2024/video/class/Image/9.png"), sg.Text("9 Sad     "),sg.Image(r"/home/rvp/Work_Student2024/video/class/Image/19.png"), sg.Text("19 Worried")],
        ]

        video_play_column = [
            [sg.Text("Video Directory")], [sg.Input(key="_FILEPATH_"), sg.Button("Browse")],
            [sg.Canvas(size=(500, 500), key="canvas", background_color="blue")],
            [sg.Slider(size=(30, 20), range=(0, 100), resolution=100, key="slider", orientation="h",
                             enable_events=True), sg.T("0", key="counter", size=(10, 1))],
            [sg.Button('Next frame'), sg.Button("Pause", key="Play"), sg.Button('Exit')]
        ]
        image_viewer_column = [
            [sg.Text("Image List")],
            [sg.Button('Load Image')],
            [sg.Column(table, background_color='black', pad=(0, 0), key='Table'),
                sg.Slider(range=(0, size-rows), size=(21, 24), orientation='v', **option, key='V_Scrollbar')],
            #[sg.Canvas(size=(200, 500), key="canvas", background_color="white")],  
            [sg.Button(button_text='Add', key='ADD_IMG')]
        ]

        layout = [
            [
                sg.Menu(menu_def),
                sg.Column(file_list_column),
                sg.Column(video_play_column, element_justification='center'),
                sg.Column(image_viewer_column, element_justification='center')                 
            ]
        ]

        self.window = sg.Window('FASHION', layout).Finalize()
        # set return_keyboard_events=True to make hotkeys for video playback
        # Get the tkinter canvas for displaying the video
        canvas = self.window.Element("canvas")
        self.canvas = canvas.TKCanvas

        # Start video display thread
        self.load_video()
        
        item_index = 1
        # items_count = 0
        
        for y in range(rows):
            for x in range(cols):
                element = self.window[(y, x)]
                element.Widget.configure(takefocus=0)
        #        element.Widget.bind('<MouseWheel>', vscroll)
        #        element.ParentRowFrame.bind('<MouseWheel>', vscroll)
        

        self.window['V_Scrollbar'].Update(value=size-rows+1)
        #self.window['V_Scrollbar'].Widget.bind('<MouseWheel>', vscroll)
        
        loaded = False
        
        while True:  # Main event Loop
            event, values = self.window.Read()

            # print(event, values)
            if event is None or event == 'Exit':
                """Handle exit"""
                break
            if event == "Browse":
                """Browse for files when the Browse button is pressed"""
                # Open a file dialog and get the file path
                video_path = None
                try:
                    video_path = sg.filedialog.askopenfile().name
                except AttributeError:
                    print("no video selected, doing nothing")

                if video_path:
                    print(video_path)
                    # Initialize video
                    self.vid = MyVideoCapture(video_path)
                    # Calculate new video dimensions
                    self.vid_width = 500
                    self.vid_height = int(self.vid_width * self.vid.height / self.vid.width)
                    # print("old par: %f" % (self.vid.width / self.vid.height))
                    # print("new par: %f" % (self.vid_width / self.vid_height))
                    # print(self.vid.fps)
                    # print(int(self.vid.frames))
                    self.frames = int(self.vid.frames)
#--------------------------------------------------------------------------------
                    #results = model.predict(self.frames)
                    #print(class_predict)
#--------------------------------------------------------------------------------
                    # Update slider to match amount of frames
                    self.window.Element("slider").Update(range=(0, int(self.frames)), value=0)
                    # Update right side of counter
                    self.window.Element("counter").Update("0/%i" % self.frames)
                    # change canvas size approx to video size
                    self.canvas.config(width=self.vid_width, height=self.vid_height)

                    # Reset frame count
                    self.frame = 0
                    self.delay = 1 / self.vid.fps

                    # Update the video path text field
                    self.window.Element("_FILEPATH_").Update(video_path)

            if event == "Play":
                if self.play:
                    self.play = False
                    self.window.Element("Play").Update("Play")
                else:
                    self.play = True
                    self.window.Element("Play").Update("Pause")

            if event == 'Next frame':
                # Jump forward a frame TODO: let user decide how far to jump
                self.set_frame(self.frame + 1)

            if event == "slider":
                # self.play = False
                self.set_frame(int(values["slider"]))
                # print(values["slider"])
                
            if event == 'ADD_IMG':
                AddImg()
                
            if event == sg.WINDOW_CLOSED:
                break
        
            elif event == 'Load Image':
                print("Load Image")
                loaded = True
                scroll()
        
            elif event in ('V_Scrollbar'):
                y0 = size - rows - int(values['V_Scrollbar'])
                scroll()
        
        # Exiting
        # print("bye :)")
        self.window.Close()
        sys.exit()




class MyVideoCapture:
    """
    Defines a new video loader with openCV
    Original code from https://solarianprogrammer.com/2018/04/21/python-opencv-show-video-tkinter-window/
    Modified by me
    """
    def __init__(self, video_source):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.vid.get(cv2.CAP_PROP_FPS)

    def get_frame(self):
        """
        Return the next frame
        """
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return 0, None
    def goto_frame(self, frame_no):
        """
        Go to specific frame
        """
        if self.vid.isOpened():
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_no)  # Set current frame
            ret, frame = self.vid.read()  # Retrieve frame
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return 0, None
    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


if __name__ == '__main__':
    App()
