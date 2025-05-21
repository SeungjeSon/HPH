import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog


class DataVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Data Visualizer")
        self.geometry("1200x800")

        # 상단 버튼 프레임
        self.button_frame = tk.Frame(self)
        self.button_frame.pack(pady=10)

        self.load_button = tk.Button(self.button_frame, text="Load CSV", command=self.load_csv)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.plot_button = tk.Button(self.button_frame, text="Plot Data", command=self.plot_selected_data)
        self.plot_button.pack(side=tk.LEFT, padx=5)

        self.zoomout_button = tk.Button(self.button_frame, text="Zoom Out", command=self.zoom_out)
        self.zoomout_button.pack(side=tk.LEFT, padx=5)

        self.coord_label = tk.Label(self, text="")
        self.coord_label.pack(pady=5)

        # 체크박스 프레임을 3열로 배치
        self.check_frame = tk.Frame(self)
        self.check_frame.pack(pady=5)

        self.check_vars = {}

        # 그래프 영역
        self.figure = plt.Figure()
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.df = None
        self.ax1 = None
        self.ax_list = []
        self.zoom_rect = None
        self.zoomed = False
        self.press_event = None
        self.rect = None

        # 범위 설정 UI
        self.range_frame = tk.Frame(self)
        self.range_frame.pack(pady=5)

        tk.Label(self.range_frame, text="X-axis Min:").pack(side=tk.LEFT, padx=5)
        self.x_min_entry = tk.Entry(self.range_frame, width=10)
        self.x_min_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(self.range_frame, text="X-axis Max:").pack(side=tk.LEFT, padx=5)
        self.x_max_entry = tk.Entry(self.range_frame, width=10)
        self.x_max_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(self.range_frame, text="Y-axis Min:").pack(side=tk.LEFT, padx=5)
        self.y_min_entry = tk.Entry(self.range_frame, width=10)
        self.y_min_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(self.range_frame, text="Y-axis Max:").pack(side=tk.LEFT, padx=5)
        self.y_max_entry = tk.Entry(self.range_frame, width=10)
        self.y_max_entry.pack(side=tk.LEFT, padx=5)

        self.set_range_button = tk.Button(self.range_frame, text="Set Range", command=self.set_custom_range)
        self.set_range_button.pack(side=tk.LEFT, padx=5)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            self.df.fillna(method='ffill', inplace=True)
            self.create_check_buttons()

    def create_check_buttons(self):
        for widget in self.check_frame.winfo_children():
            widget.destroy()

        columns_to_plot = self.df.columns[1:]  # 첫 번째 열을 제외한 모든 열 선택
        self.check_vars = {col: tk.BooleanVar() for col in columns_to_plot}

        # 3열로 배치
        rows = (len(columns_to_plot) + 2) // 3
        for i, column in enumerate(columns_to_plot):
            check_button = tk.Checkbutton(self.check_frame, text=column, variable=self.check_vars[column])
            check_button.grid(row=i % rows, column=i // rows, sticky=tk.W, padx=5, pady=2)

    def downsample(self, data, factor=10):
        return data[::factor]

    def plot_selected_data(self):
        selected_columns = [col for col, var in self.check_vars.items() if var.get()]
        if selected_columns and self.df is not None:
            self.figure.clear()
            self.ax_list = []

            self.ax1 = self.figure.add_subplot(111)
            self.ax_list.append(self.ax1)

            colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'yellow',
                      'black', 'gold', 'lime', 'indigo']
            for i, column in enumerate(selected_columns):
                ax = self.ax1 if i == 0 else self.ax1.twinx()
                if i != 0:
                    ax.spines["right"].set_position(("outward", 60 * (i - 1)))
                    self.ax_list.append(ax)
                downsampled_data = self.downsample(self.df[column].values)
                downsampled_index = self.downsample(self.df.index.values)
                ax.plot(downsampled_index, downsampled_data, label=column, color=colors[i % len(colors)])
                ax.set_ylabel(column)

            self.ax1.set_xlabel('Index')
            self.ax1.legend(loc='upper left')
            self.canvas.draw()
            self.connect_zoom()

    def connect_zoom(self):
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def set_custom_range(self):
        try:
            x_min = float(self.x_min_entry.get())
            x_max = float(self.x_max_entry.get())
            y_min = float(self.y_min_entry.get())
            y_max = float(self.y_max_entry.get())
            for ax in self.ax_list:
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
            self.canvas.draw()
        except ValueError:
            self.coord_label.config(text="Invalid range values. Please enter numeric values.")

    def zoom_out(self):
        if self.df is not None:
            for ax in self.ax_list:
                ax.set_xlim(self.df.index.min(), self.df.index.max())
                ax.set_ylim(self.df.min().min(), self.df.max().max())
            self.canvas.draw()


if __name__ == "__main__":
    app = DataVisualizer()
    app.mainloop()
