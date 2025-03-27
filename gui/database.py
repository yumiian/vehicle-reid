import sqlite3
import os
from pathlib import Path
import pandas as pd
import streamlit as st

import settings

db_path = Path(settings.DATABASE_FILEPATH)
backup_db_path = Path(settings.BACKUP_DB_FILEPATH)

@st.cache_resource
def init_db_connection():
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return conn

def create_table(table):
    conn = init_db_connection()
    cursor = conn.cursor()

    cursor.execute("PRAGMA foreign_keys = ON;")

    if table == "checkpoint":
        query = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id INTEGER PRIMARY KEY,
                image1 CHAR(6) NOT NULL,
                image2 CHAR(6) NOT NULL,
                filepath1 TEXT NOT NULL,
                filepath2 TEXT NOT NULL,
                UNIQUE(image1, image2)
                )
        """
    elif table == "comparison":
        cursor.execute(f"DROP TABLE IF EXISTS {table};")

        query = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id INTEGER PRIMARY KEY,
                filepath1 TEXT,
                filepath2 TEXT
                )
        """
    elif table == "saved":
        cursor.execute(f"DROP TABLE IF EXISTS {table};")

        query = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id INTEGER PRIMARY KEY,
                image1 CHAR(6) NOT NULL,
                image2 CHAR(6) NOT NULL
                )
        """
    elif table == "video":
        query = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                video_id INTEGER PRIMARY KEY,
                location VARCHAR(6) NOT NULL,
                camera_id VARCHAR(6) NOT NULL,
                time VARCHAR(6) NOT NULL
                )
        """
    elif table == "image":
        query = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                image_id INTEGER PRIMARY KEY,
                video_id INTEGER NOT NULL,
                frame_id CHAR(6) NOT NULL,
                FOREIGN KEY (video_id) REFERENCES video(video_id) ON DELETE CASCADE ON UPDATE CASCADE
                )
        """
    elif table == "crop_image":
        query = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                crop_id INTEGER PRIMARY KEY,
                image_id INTEGER NOT NULL,
                label_id CHAR(6) NOT NULL,
                FOREIGN KEY (image_id) REFERENCES image(image_id) ON DELETE CASCADE ON UPDATE CASCADE
                )
        """
    else:
        raise ValueError("Invalid table name.")

    cursor.execute(query)

    conn.commit()

def get_last_id(table, id_name):
    conn = init_db_connection()
    cursor = conn.cursor()

    cursor.execute(f"SELECT MAX({id_name}) FROM {table}")
    result = cursor.fetchone()[0] # last row id
    last_id = (int(result) + 1) if result is not None else 1  # Avoid NULL issue, id starts at 1

    return last_id

def insert_data(table, data1=None, data2=None):
    conn = init_db_connection()
    cursor = conn.cursor()

    if table == "checkpoint":
        last_id = get_last_id(table, "id")
        query = f"INSERT INTO {table} (id, image1, image2, filepath1, filepath2) VALUES (?, ?, ?, ?, ?)"

        image1 = os.path.splitext(st.session_state.image_list1[st.session_state.img1])[0].split("-")[-1]
        image2 = os.path.splitext(st.session_state.image_list2[st.session_state.img2])[0].split("-")[-1]
        filepath1 = st.session_state.image_list1[st.session_state.img1]
        filepath2 = st.session_state.image_list2[st.session_state.img2]

        cursor.execute(query, (last_id, image1, image2, filepath1, filepath2))
        
        st.session_state.checkpoint_id = last_id

    elif table == "comparison":
        last_id = get_last_id(table, "id")
        query = f"INSERT INTO {table} (id, filepath1, filepath2) VALUES (?, ?, ?)"

        max_length = max(len(data1), len(data2))
        # Extend the shorter list with None values
        data1.extend([None] * (max_length - len(data1)))
        data2.extend([None] * (max_length - len(data2)))

        # Create tuples with unique IDs
        rows = [(last_id + i, d1, d2) for i, (d1, d2) in enumerate(zip(data1, data2))]

        cursor.executemany(query, rows)

    elif table == "video":
        last_id = get_last_id(table, "video_id")
        query = f"INSERT INTO {table} (video_id, location, camera_id, time) VALUES (?, ?, ?, ?)"

        cursor.execute(query, (last_id, st.session_state.location, st.session_state.camera_id, st.session_state.time))
        st.session_state.video_id = last_id

    elif table == "image":
        last_id = get_last_id(table, "image_id")
        query = f"INSERT INTO {table} (image_id, video_id, frame_id) VALUES (?, ?, ?)"

        cursor.execute(query, (last_id, st.session_state.video_id, data1))
        st.session_state.image_id = last_id

    elif table == "crop_image":
        last_id = get_last_id(table, "crop_id")
        query = f"INSERT INTO {table} (crop_id, image_id, label_id) VALUES (?, ?, ?)"

        cursor.execute(query, (last_id, st.session_state.image_id, data1))
        
    else:
        raise ValueError("Invalid table name.")
    
    conn.commit()

def undo_last_match(table):
    conn = init_db_connection()
    cursor = conn.cursor()

    query = f"DELETE FROM {table} WHERE rowid = ?"
    cursor.execute(query, (st.session_state.checkpoint_id,)) # tuple with comma
    
    conn.commit()

def compare_data(table_more, table_less, column_name):
    conn = init_db_connection()
    conn.row_factory = sqlite3.Row # access data by column name
    cursor = conn.cursor()

    query = f"""
        SELECT * FROM {table_more}
        WHERE {column_name} NOT IN (SELECT {column_name} FROM {table_less});
    """

    cursor.execute(query)
    results = cursor.fetchall()

    return results

def select_data(table, column_name):
    conn = init_db_connection()
    cursor = conn.cursor()

    query = f"SELECT {column_name} FROM {table};"

    cursor.execute(query)
    results = cursor.fetchall()

    return results

def check_table_exist(table):
    conn = init_db_connection()
    cursor = conn.cursor()

    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}';")
    exist = cursor.fetchone()

    return exist

def check_table_empty(table):
    conn = init_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(f"SELECT COUNT(*) FROM {table};")
    row_count = cursor.fetchone()[0]

    if row_count == 0:
        return True
    
    return False
    
def drop_table(table):
    conn = init_db_connection()
    cursor = conn.cursor()

    cursor.execute(f"DROP TABLE IF EXISTS {table};")

    conn.commit()

def copy_table(source_table, dest_table, data):
    conn = init_db_connection()
    cursor = conn.cursor()

    # cursor.execute(f"CREATE TABLE {dest_table} AS SELECT * FROM {source_table}")

    cursor.execute(f'''
        INSERT INTO {dest_table} ({data})
        SELECT {data} FROM {source_table}
    ''')

    conn.commit()

def get_table_names():
    conn = init_db_connection() 
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence';")  
    tables = [row[0] for row in cursor.fetchall()]
    
    return tables

def db_to_df(table):
    conn = init_db_connection()

    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)

    conn.commit()

    return df

def df_to_db(df, table):
    conn = init_db_connection()

    df.to_sql(name=table, con=conn, if_exists="replace", index=False)

@st.dialog("Confirm delete database?")
def dialog_delete_db():
    input = st.text_input('Are you sure you want to delete the database? Type "YES" to confirm.')
    if st.button("Submit"):
        if input != "YES":
            st.error("Invalid input!")
            return
        
        if os.path.exists(db_path):
            os.remove(db_path)
        else:
            st.error("Database file not found or has already been deleted!")
            return

        st.success("Database successfully deleted. Please refresh the page.")

def backup():
    source_conn = init_db_connection()
    backup_conn = sqlite3.connect(backup_db_path)

    with backup_conn:
        source_conn.backup(backup_conn)

    st.success("Database backup successfully!")