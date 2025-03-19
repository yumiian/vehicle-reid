import sqlite3
import os
import streamlit as st

def create_table(table):
    conn = sqlite3.connect("gui/reid.db")
    cursor = conn.cursor()

    if table == "checkpoint":
        query = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image1 CHAR(6) NOT NULL,
                image2 CHAR(6) NOT NULL,
                filepath1 TEXT NOT NULL,
                filepath2 TEXT NOT NULL
                )
        """
    elif table == "comparison":
        # drop the table if it exists
        cursor.execute(f"DROP TABLE IF EXISTS {table};")

        query = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath1 TEXT NOT NULL,
                filepath2 TEXT NOT NULL
                )
        """
    elif table == "saved":
        # drop the table if it exists
        cursor.execute(f"DROP TABLE IF EXISTS {table};")

        query = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image1 CHAR(6) NOT NULL,
                image2 CHAR(6) NOT NULL
                )
        """
    else:
        raise ValueError("Invalid table name.")

    cursor.execute(query)

    conn.commit()
    conn.close()

def insert_data(table, data=None):
    conn = sqlite3.connect("gui/reid.db")
    cursor = conn.cursor()

    # if overwrite:
    #     # clear the table if it is not empty
    #     cursor.execute(f"SELECT COUNT(*) FROM {table};")
    #     row_count = cursor.fetchone()[0]
    #     if row_count > 0:
    #         cursor.execute(f"DELETE FROM {table};")

    if table == "checkpoint":
        query = f"INSERT INTO {table} (image1, image2, filepath1, filepath2) VALUES (?, ?, ?, ?)"

        image1 = os.path.splitext(st.session_state.image_list1[st.session_state.img1])[0].split("-")[-1]
        image2 = os.path.splitext(st.session_state.image_list2[st.session_state.img2])[0].split("-")[-1]
        filepath1 = st.session_state.image_list1[st.session_state.img1]
        filepath2 = st.session_state.image_list2[st.session_state.img2]

        cursor.execute(query, (image1, image2, filepath1, filepath2))
    elif table == "comparison":
        query = f"INSERT INTO {table} (filepath1, filepath2) VALUES (?, ?)"

        cursor.executemany(query, data)
    else:
        raise ValueError("Invalid table name.")

    conn.commit()
    conn.close()

def compare_data(table_more, table_less, column_name):
    conn = sqlite3.connect("gui/reid.db")
    conn.row_factory = sqlite3.Row # access data by column name
    cursor = conn.cursor()

    query = f"""
        SELECT * FROM {table_more}
        WHERE {column_name} NOT IN (SELECT {column_name} FROM {table_less});
    """

    cursor.execute(query)
    results = cursor.fetchall()

    conn.close()

    return results

def select_data(table, column_name):
    conn = sqlite3.connect("gui/reid.db")
    cursor = conn.cursor()

    query = f"SELECT {column_name} FROM {table}"

    cursor.execute(query)
    results = cursor.fetchall()

    conn.close()

    return results

def check_table_exist(table):
    conn = sqlite3.connect("gui/reid.db")
    cursor = conn.cursor()

    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}';")
    exist = cursor.fetchone()

    conn.close()

    return exist

def drop_table(table):
    conn = sqlite3.connect("gui/reid.db")
    cursor = conn.cursor()

    cursor.execute(f"DROP TABLE IF EXISTS {table};")

    conn.commit()
    conn.close()

def copy_table(source_table, dest_table, data):
    conn = sqlite3.connect("gui/reid.db")
    cursor = conn.cursor()

    # cursor.execute(f"CREATE TABLE {dest_table} AS SELECT * FROM {source_table}")

    cursor.execute(f'''
        INSERT INTO {dest_table} ({data})
        SELECT {data} FROM {source_table}
    ''')

    conn.commit()
    conn.close()