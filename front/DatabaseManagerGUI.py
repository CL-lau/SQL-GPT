import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from sqlalchemy import create_engine, inspect


class DatabaseManagerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("数据库管理工具")

        self.db_urls = {
            "database1": "sqlite:///database1.db",
            "database2": "sqlite:///database2.db"
        }

        self.db_manager = DatabaseManager(self.db_urls)

        self.db_select_label = tk.Label(self.root, text="选择数据库:")
        self.db_select_label.pack()

        self.db_combo = ttk.Combobox(self.root, values=list(self.db_urls.keys()))
        self.db_combo.pack()

        self.table_label = tk.Label(self.root, text="表格:")
        self.table_label.pack()

        self.table_listbox = tk.Listbox(self.root)
        self.table_listbox.pack()

        self.column_label = tk.Label(self.root, text="列名:")
        self.column_label.pack()

        self.column_listbox = tk.Listbox(self.root)
        self.column_listbox.pack()

        self.query_label = tk.Label(self.root, text="SQL 查询:")
        self.query_label.pack()

        self.query_text = tk.Text(self.root, height=5, width=50)
        self.query_text.pack()

        self.execute_button = tk.Button(self.root, text="执行查询", command=self.execute_query)
        self.execute_button.pack()

        self.root.mainloop()

    def update_table_list(self):
        self.table_listbox.delete(0, tk.END)
        tables = self.db_manager.get_table_list(self.db_combo.get())
        for table in tables:
            self.table_listbox.insert(tk.END, table)

    def update_column_list(self):
        self.column_listbox.delete(0, tk.END)
        selected_table = self.table_listbox.get(tk.ACTIVE)
        columns = self.db_manager.get_column_list(self.db_combo.get(), selected_table)
        for column in columns:
            self.column_listbox.insert(tk.END, column)

    def execute_query(self):
        selected_db = self.db_combo.get()
        query = self.query_text.get("1.0", tk.END).strip()

        if query:
            try:
                results = self.db_manager.execute_query(selected_db, query)
                messagebox.showinfo("查询结果", str(results))
            except Exception as e:
                messagebox.showerror("错误", f"查询错误：{e}")


class DatabaseManager:
    def __init__(self, db_urls):
        self.engines = {db_name: create_engine(url) for db_name, url in db_urls.items()}
        self.inspector = inspect(self.engines[next(iter(self.engines))])  # Use the first engine to inspect tables

    def get_table_list(self, db_name):
        self.inspector.bind = self.engines[db_name]
        return self.inspector.get_table_names()

    def get_column_list(self, db_name, table_name):
        self.inspector.bind = self.engines[db_name]
        columns = self.inspector.get_columns(table_name)
        return [column['name'] for column in columns]

    def execute_query(self, db_name, query):
        engine = self.engines[db_name]
        with engine.connect() as connection:
            result = connection.execute(query)
            return result.fetchall()


if __name__ == "__main__":
    root = tk.Tk()
    app = DatabaseManagerGUI(root)
