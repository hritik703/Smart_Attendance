import streamlit as st
import pandas as pd
import cv2
import numpy as np
import json
import re
from datetime import date, datetime, timedelta
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

from database import (
    get_all_faces, get_all_users, get_today_record, 
    mark_check_in, mark_check_out, add_user, add_face, 
    get_all_attendance, delete_user
)
from core import get_embedding_from_image

st.set_page_config(page_title="AttendAI Web Terminal", layout="wide")

# ── Dynamic Configuration Sidebar ──
st.sidebar.title("🛠️ Session Configuration")
CLASS_NAME = st.sidebar.text_input("Active Class Name", value="General")
CHECKOUT_TIME = st.sidebar.text_input("Checkout Rule / Gap", value="+1h00m")

def get_minimum_gap_seconds(checkout_str: str) -> int:
    if checkout_str.startswith("+"):
        h = int(re.search(r"(\d+)h", checkout_str).group(1)) if re.search(r"(\d+)h", checkout_str) else 0
        m = int(re.search(r"(\d+)m", checkout_str).group(1)) if re.search(r"(\d+)m", checkout_str) else 0
        return (h * 3600) + (m * 60)
    return 0

MIN_CHECKIN_GAP = get_minimum_gap_seconds(CHECKOUT_TIME)

# ── Multi-Mode Web Routing ──
menu = ["📊 Attendance Dashboard", "📷 Check-In/Out Station", "👤 Student Registration"]
choice = st.sidebar.selectbox("Navigate Station", menu)

# Cache system coordinates for high performance live feeds
@st.cache_data(ttl=10)
def load_system_identities():
    face_rows = get_all_faces()
    user_rows = get_all_users()
    names = {str(u['user_id']): u['full_name'] for u in user_rows}
    known_ids = [str(r['user_id']) for r in face_rows]
    known_encs = [np.array(json.loads(r['embedding'])) for r in face_rows]
    return names, known_ids, known_encs

# ── ROUTE 1: DASHBOARD ──
if choice == "📊 Attendance Dashboard":
    st.title("AttendAI Control Management Dashboard")
    try:
        records = get_all_attendance()
        if records:
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True)
            
            # Action: Remove student
            st.subheader("Manage Database Context")
            del_id = st.text_input("Enter Student ID to fully delete:")
            if st.button("Delete Identity Profile") and del_id:
                delete_user(del_id.strip())
                st.success(f"Successfully detached Profile {del_id}")
                st.rerun()
        else:
            st.info("No logs present in cloud registry.")
    except Exception as e:
        st.error(f"Error communicating with analytical database instance: {e}")

# ── ROUTE 2: MONITOR FEED (ATTENDANCE CAM) ──
elif choice == "📷 Check-In/Out Station":
    st.title(f"Live Terminal Feed — {CLASS_NAME}")
    st.info(f"Target Policy Window: Checks restricted until {MIN_CHECKIN_GAP // 60}m active duration.")
    
    names, known_ids, known_encs = load_system_identities()
    
    class AttendanceProcessor(VideoTransformerBase):
        def __init__(self):
            self.cooldown = {}
            self.face_status = {}

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            SCALE = 0.25
            small = cv2.resize(img, (0, 0), fx=SCALE, fy=SCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            
            import face_recognition
            locs = face_recognition.face_locations(rgb_small, model="hog")
            encs = face_recognition.face_encodings(rgb_small, locs)
            
            for enc, loc in zip(encs, locs):
                name = "Unknown"
                color = (0, 0, 255)
                label = ""
                
                if known_encs:
                    dists = face_recognition.face_distance(known_encs, enc)
                    best_idx = int(np.argmin(dists))
                    if dists[best_idx] < 0.5:
                        uid = str(known_ids[best_idx])
                        name = names.get(uid, "Unknown")
                        
                        if name != "Unknown":
                            color = (0, 250, 0)
                            now = datetime.now()
                            last = self.cooldown.get(uid)
                            
                            if last is None or (now - last).total_seconds() > 15:
                                self.cooldown[uid] = now
                                t = now.strftime('%H:%M:%S')
                                existing = get_today_record(uid, CLASS_NAME)
                                
                                if existing is None:
                                    mark_check_in(uid, name, CLASS_NAME)
                                    self.face_status[uid] = f"CHECK-IN {t}"
                                elif existing.get('check_out') == '':
                                    checkin_dt = datetime.combine(date.today(), datetime.strptime(str(existing['check_in']), '%H:%M:%S').time())
                                    gap_seconds = (now - checkin_dt).total_seconds()
                                    
                                    if gap_seconds >= MIN_CHECKIN_GAP:
                                        mark_check_out(uid, CLASS_NAME)
                                        self.face_status[uid] = f"CHECK-OUT {t}"
                                    else:
                                        rem = int(MIN_CHECKIN_GAP - gap_seconds)
                                        self.face_status[uid] = f"Wait {rem // 60}m {rem % 60}s"
                                else:
                                    self.face_status[uid] = "Completed Today"
                        label = self.face_status.get(uid, "")
                
                top, right, bottom, left = [int(v / SCALE) for v in loc]
                cv2.rectangle(img, (left, top), (right, bottom), color, 2)
                if label:
                    cv2.putText(img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            return img

    webrtc_streamer(key="attendance", mode=WebRtcMode.VIDEO_RECVONLY, video_transformer_factory=AttendanceProcessor)

# ── ROUTE 3: REGISTRATION PANEL ──
elif choice == "👤 Student Registration":
    st.title("Register Target Individual Context")
    
    reg_name = st.text_input("Full Name:")
    reg_id = st.text_input("Unique Student ID:")
    
    class RegistrationProcessor(VideoTransformerBase):
        def __init__(self):
            self.last_frame = None
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.last_frame = img.copy()
            return img

    ctx = webrtc_streamer(key="register", mode=WebRtcMode.VIDEO_RECVONLY, video_transformer_factory=RegistrationProcessor)
    
    if st.button("Capture and Register Geometric Vector Profile"):
        if reg_name and reg_id and ctx.video_transformer and ctx.video_transformer.last_frame is not None:
            try:
                target_img = ctx.video_transformer.last_frame
                embedding = get_embedding_from_image(target_img)
                
                add_user(reg_name, reg_id)
                add_face(reg_id, json.dumps(embedding))
                st.success(f"Successfully mapped profile matrix for {reg_name} (ID: {reg_id})!")
            except Exception as e:
                st.error(f"Processing Matrix Exception: {e}. Ensure face is fully centered.")
        else:
            st.warning("Ensure fields are populated and the camera stream is running.")
