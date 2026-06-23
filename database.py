import os
import json
import numpy as np
import mysql.connector
from datetime import date, datetime

def get_conn():
    """Dynamically reads credentials from Streamlit Secrets or defaults locally."""
    import streamlit as st
    if "db" in st.secrets:
        return mysql.connector.connect(
            host=st.secrets["db"]["host"],
            user=st.secrets["db"]["user"],
            password=st.secrets["db"]["password"],
            database=st.secrets["db"]["database"],
            port=st.secrets["db"].get("port", 3306)
        )
    else:
        # Local fallback if testing on a native machine
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="password",
            database="face_recog_attendence"
        )

def get_all_faces():
    conn = get_conn()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT user_id, face_encoding as embedding FROM face_data")
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_all_users():
    conn = get_conn()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT user_id, full_name FROM users")
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_today_record(user_id, class_name):
    conn = get_conn()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM attendance WHERE user_id=%s AND date=%s AND class_name=%s",
        (str(user_id), str(date.today()), class_name)
    )
    row = cursor.fetchone()
    conn.close()
    return row

def mark_check_in(user_id, name, class_name):
    conn = get_conn()
    cursor = conn.cursor()
    t = datetime.now().strftime('%H:%M:%S')
    d = str(date.today())
    cursor.execute(
        "INSERT INTO attendance (user_id, full_name, date, check_in, check_out, class_name) VALUES (%s, %s, %s, %s, '', %s)",
        (str(user_id), name, d, t, class_name)
    )
    conn.commit()
    conn.close()

def mark_check_out(user_id, class_name):
    conn = get_conn()
    cursor = conn.cursor()
    t = datetime.now().strftime('%H:%M:%S')
    cursor.execute(
        "UPDATE attendance SET check_out=%s WHERE user_id=%s AND date=%s AND class_name=%s AND check_out=''",
        (t, str(user_id), str(date.today()), class_name)
    )
    conn.commit()
    conn.close()

def add_user(name, student_id):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (user_id, full_name) VALUES (%s, %s)", (str(student_id), name))
    conn.commit()
    conn.close()

def add_face(student_id, embedding_json):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO face_data (user_id, face_encoding) VALUES (%s, %s)", (str(student_id), embedding_json))
    conn.commit()
    conn.close()

def delete_user(student_id):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE user_id=%s", (str(student_id),))
    cursor.execute("DELETE FROM face_data WHERE user_id=%s", (str(student_id),))
    conn.commit()
    conn.close()

def get_all_attendance():
    conn = get_conn()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM attendance")
    rows = cursor.fetchall()
    conn.close()
    return rows
