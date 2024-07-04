from aiohttp import request
from flask import Flask, redirect, render_template, Response, jsonify, session, url_for, request, url_for, flash
import cv2
import dlib
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from imutils import face_utils
import time
import pyttsx3
import winsound
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from flask_socketio import SocketIO
from werkzeug.security import generate_password_hash, check_password_hash
import os
from sqlalchemy.exc import IntegrityError 

app = Flask(__name__)

# Configure SQLAlchemy
engine = create_engine('sqlite:///your_database.db', echo=True)
Base = declarative_base()

class DetectionEvent(Base):
    __tablename__ = 'detection_events'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    event_type = Column(String)

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
socketio = SocketIO(app)

##user Authentication 
# Get the current directory of the script
basedir = os.path.abspath(os.path.dirname(__file__))

# Use the current directory to create the absolute path for the database file
db_path = os.path.join(basedir, 'your_database.db')
app.config['SECRET_KEY'] = '@#$%^&*()'  
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path
db = SQLAlchemy(app)
migrate = Migrate(app, db)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(100), nullable=False)
    
#THRESHOLD values
EAR_THRESHOLD = 0.25
YAWN_VERTICAL_THRESHOLD = 33
YAWN_CONSECUTIVE_FRAMES_THRESHOLD =20
EYES_CLOSED_DURATION_THRESHOLD = 3.0
EYES_OPEN_DURATION_THRESHOLD = 15

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\LENOVO\\Desktop\\shape_predictor_68_face_landmarks.dat")

blink_counter = 0
yawn_counter = 0
eyes_closed_timer_start = None
eyes_open_timer_start = None

def speak_message(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

def eye_aspect_ratio(eye):
    A = cv2.norm(eye[1] - eye[5])
    B = cv2.norm(eye[2] - eye[4])
    C = cv2.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def play_alert_audio():
    winsound.Beep(1000, 500)  # Play a beep sound for alert

def detect_drowsiness(frame, session):
    global blink_counter, yawn_counter, eyes_closed_timer_start, eyes_open_timer_start

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

    faces = detector(gray)
    for face in faces:
        landmarks = face_utils.shape_to_np(predictor(gray, face))
        mouth_top = landmarks[51][1]  
        mouth_bottom = landmarks[57][1]  
        eye_left = landmarks[42:48]
        eye_right = landmarks[36:42]

        left_ear = eye_aspect_ratio(eye_left)
        right_ear = eye_aspect_ratio(eye_right)
        average_ear = (left_ear + right_ear) / 2.0
        
        # Detect closed eyes
        if average_ear < EAR_THRESHOLD:
            blink_counter += 1
            if eyes_closed_timer_start is None:
                eyes_closed_timer_start = time.time()
                if eyes_open_timer_start is not None:
                    eyes_open_timer_start = None
        else:
            blink_counter = 0
            eyes_closed_timer_start = None

        # Detect continuous open eyes
        if average_ear >= EAR_THRESHOLD:
            if eyes_open_timer_start is None:
                eyes_open_timer_start = time.time()
            elif time.time() - eyes_open_timer_start > EYES_OPEN_DURATION_THRESHOLD:
               
                play_alert_audio()
                cv2.putText(frame, "Alert: Eyes open for too long", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                speak_message("Alert: Eyes open for too long !")

        # Detect yawn
        yawn_vertical_distance = mouth_bottom - mouth_top
        if yawn_vertical_distance > YAWN_VERTICAL_THRESHOLD:
            yawn_counter += 1
            if yawn_counter >= YAWN_CONSECUTIVE_FRAMES_THRESHOLD:
                cv2.putText(frame, "Yawning Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                session.add(DetectionEvent(event_type="Yawning Detected"))
                session.commit()
               
                speak_message("Alert: Yawning detected !")
        else:
            yawn_counter = 0

        # Detect sleepy eyes
        if eyes_closed_timer_start is not None and time.time() - eyes_closed_timer_start > EYES_CLOSED_DURATION_THRESHOLD:
            session.add(DetectionEvent(event_type="Eyes Closed (sleepy)"))
            session.commit()
            cv2.putText(frame, "Sleepy", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            play_alert_audio()
            
            if eyes_open_timer_start is not None:
                eyes_open_timer_start = None

        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    return frame


def gen_frames_and_save():  
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            session = Session()
            frame = detect_drowsiness(frame, session)
            session.close()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
@app.route('/')
def mainf():
    return render_template('main.html')

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about_us.html')

# Define the model for the contact form data
class ContactMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Save the form data to the database
        new_message = ContactMessage(name=name, email=email, message=message)
        db.session.add(new_message)
        db.session.commit()

    return render_template('contact.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames_and_save(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/latest_statistics')
def latest_statistics():
    session = Session()
    yawning_events, sleepy_events, total_events = calculate_statistics(session)
    session.close()
    return jsonify({
        'yawning_events': yawning_events,
        'sleepy_events': sleepy_events,
        'total_events': total_events
    })
    
@socketio.on('update_chart')
def handle_update_chart(yawning_events, sleepy_events):
    socketio.emit('update_chart', {'yawning_events': yawning_events, 'sleepy_events': sleepy_events}, broadcast=True)


def calculate_statistics(session):
    yawning_events = session.query(DetectionEvent).filter_by(event_type='Yawning Detected').count()
    sleepy_events = session.query(DetectionEvent).filter_by(event_type='Eyes Closed (sleepy)').count()
    total_events = session.query(DetectionEvent).count()
    return yawning_events, sleepy_events, total_events

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        Name = request.form['name']
        Email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        user = User(name=Name, email=Email, password_hash=hashed_password)
        try:
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except IntegrityError:
            db.session.rollback()  # Roll back the session to avoid partial changes
            flash('Email already exists. Please use a different email.', 'danger')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            flash('Logged in successfully!', 'success')
            session['user_name'] = user.name  # Store user's name in session
            session['user_id'] = user.id
            # Query for all records in the table
            db_session = Session()
            all_records = db_session.query(DetectionEvent).all()

            # Delete all records
            for record in all_records:
                db_session.delete(record)

            # Commit the transaction
            db_session.commit()
            db_session.close()
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('login.html')

def create_db():
    with app.app_context():
        db.create_all()
        
if __name__ == '__main__':
    create_db()
    app.run(debug=True)
    socketio.run(app, debug=True)