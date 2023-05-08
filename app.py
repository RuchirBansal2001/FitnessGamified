import tempfile
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import time
from PIL import Image
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

DEMO_IMAGE = 'demo.jpg'
DEMO_VIDEO = 'video.mp4'

st.title('FitnessGamified App')
st.markdown(
    """
    <style>
    [data-testid = "stsidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid = "stsidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    
    """,
    unsafe_allow_html= True,
)

st.sidebar.title('Fitness Gamified')
st.sidebar.subheader('parameters')

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = width/float(w)
        dim = (int(w*r), height)

    else:
        r = width/float(w)
        dim = (width, int(h*r))

    #resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized

app_mode = st.sidebar.selectbox('Choose the App mode' , ['About App', 'Dino'])
# 'Run on Image', 'Run on Video',

if app_mode == 'About App':
    st.markdown('In this application we  are using **mediapipe** for creating a Pose Detection Fitness Game.')
    st.markdown(
        """
        <style>
        [data-testid = "stsidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid = "stsidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
    
        """,
        unsafe_allow_html=True,
    )
    # st.video('https://www.youtube.com/watch?v=LL3bDpVIbOE')

    st.markdown(
        '''
        Hey **Ayush** Here! from **AryaxDD**.\n
        
        Checkcout my AMVs...Subscribe if u like\n
        
        My Socials:
        - [Youtube]('https://www.youtube.com/watch?v=LL3bDpVIbOE')
        - [LinkedIn]('https://www.linkedin.com/in/ayush-arya-b4b2331b9/')
        - [Github]('https://github.com/stack-queue-coder')
        
        '''
    )

elif app_mode == 'Run on Image':
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid = "stsidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid = "stsidebar"][aria-expanded="false"] > div:first-child{
            width: 350p
            margin-left: -350px
        }
        </style>

        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Detected Faces**")
    kpi1_text = st.markdown("0")

    max_faces = st.sidebar.number_input('maximum number of faces', value=1, min_value=1)
    st.sidebar.markdown("---")
    detection_confidence = st.sidebar.slider('min detection confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown("---")

    img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=['jpeg', 'jpg', 'png'])
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)

    face_count = 0

    with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=max_faces,
    min_detection_confidence=detection_confidence) as face_mesh:

        results = face_mesh.process(image)
        out_image = image.copy()

        for face_landmarks in results.multi_face_landmarks:
            face_count += 1

            #print('face_landmarks:', face_landmarks)

            mp_drawing.draw_landmarks(
            image=out_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
        st.subheader('Output Image')
        st.image(out_image, use_column_width=True)


elif app_mode == 'Run on Video':

    st.set_option('deprecation.showfileUploaderEncoding' ,  False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox('Record Video')

    if record:
        st.checkbox('Recording' , value=True)

    st.markdown(
        """
        <style>
        [data-testid = "stsidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid = "stsidebar"][aria-expanded="false"] > div:first-child{
            width: 350p
            margin-left: -350px
        }
        </style>

        """,
        unsafe_allow_html=True,
    )

    max_faces = st.sidebar.number_input('maximum number of faces', value=1, min_value=1)
    st.sidebar.markdown("---")
    detection_confidence = st.sidebar.slider('min detection confidence', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider('min tracking confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown("---")

    st.markdown('## Output')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO

    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    #Recording part
    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    out = cv2.VideoWriter('output1.webm', codec, fps, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)

    fps = 0
    i = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown('**Frame Rate**')
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown('**No of Faces**')
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown('**Width**')
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>" , unsafe_allow_html=True)

    with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=max_faces,
    min_detection_confidence=detection_confidence,
    min_tracking_confidence=tracking_confidence) as face_mesh:

        prevTime = 0

        while vid.isOpened():
            i +=1
            ret, frame = vid.read()
            if not ret:
                continue

            results = face_mesh.process(frame)
            frame.flags.writeable = True

            face_count = 0
            if results.multi_face_landmarks:
                #Drawing
                for face_landmarks in results.multi_face_landmarks:
                    face_count +=1

                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
            #fps logic
            currTime = time.time()
            fps = 1/(currTime - prevTime)
            prevTime = currTime

            if record:
                out.write(frame)

            #Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

            frame = cv2.resize(frame, (0,0), fx = 0.8, fy= 0.8)
            frame = image_resize(image = frame, width = 640)
            stframe.image(frame,  channels='BGR', use_column_width=True)
        # st.subheader('Output Image')
elif app_mode == 'Dino':

    st.set_option('deprecation.showfileUploaderEncoding' ,  False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox('Record Video')

    if record:
        st.checkbox('Recording' , value=True)

    st.markdown(
        """
        <style>
        [data-testid = "stsidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid = "stsidebar"][aria-expanded="false"] > div:first-child{
            width: 350p
            margin-left: -350px
        }
        </style>

        """,
        unsafe_allow_html=True,
    )

    # max_faces = st.sidebar.number_input('maximum number of faces', value=1, min_value=1)
    # st.sidebar.markdown("---")
    detection_confidence = st.sidebar.slider('min detection confidence', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider('min tracking confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown("---")

    st.markdown('## Output')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO


    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    #Recording part
    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    out = cv2.VideoWriter('output1.webm', codec, fps, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)

    fps = 0
    i = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown('**Frame Rate**')
        kpi1_text = st.markdown("0")

    with kpi3:
        st.markdown('**Width**')
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>" , unsafe_allow_html=True)

    html_string ='''
        <embed src="https://chromedino.com/" style="width:700px; height: 300px; border:1px solid black;">
    '''
    # html_string = '''
    #     <iframe src="https://chromedino.com/" frameborder="0" scrolling="no" width="100%" height="100%" loading="lazy"></iframe>
    #     <style type="text/css">iframe { position: absolute; width: 100%; height: 100%; z-index: 999; }</style>
    # '''
    # components.html(html_string)
    st.markdown(html_string, unsafe_allow_html=True)

    with mp_pose.Pose(
    min_detection_confidence=detection_confidence,min_tracking_confidence=tracking_confidence) as pose:
        prevTime = 0
        while vid.isOpened():
            i +=1
            ret, frame = vid.read()
            if not ret:
                continue

            results = pose.process(frame)
            frame.flags.writeable = True

            # extraction
            landmarks = []
            try:
                landmarks = results.pose_landmarks.landmark
                # print(landmarks)
                if landmarks[24].y <= 0.4:
                    print("up")
                    pyautogui.keyDown("space")
                    pyautogui.keyUp("space")
                elif landmarks[24].y >= 0.6:
                    print("down")
                    pyautogui.keyDown("down")
                    pyautogui.keyUp("down")
                else:
                    print("normal")
            except:
                pass

            #Drawing

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
            #fps logic
            currTime = time.time()
            fps = 1/(currTime - prevTime)
            prevTime = currTime

            if record:
                out.write(frame)

            #Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

            frame = cv2.resize(frame, (0,0), fx = 0.8, fy= 0.8)
            frame = cv2.resize(frame, (400 , 300) , interpolation = cv2.INTER_AREA)
            # frame = cv2.line(frame, (0,0.3) , (1,0.3), (0, 255, 0) , 2)
            frame = image_resize(image = frame, width = 420)
            stframe.image(frame,  channels='BGR', use_column_width=True)





