import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
from matplotlib.colors import hsv_to_rgb


class DataVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Data Visualizer")
        self.geometry("1200x800")

        # 데이터 저장소
        self.dataframes = {}  # 파일명: DataFrame 형태로 저장
        self.file_colors = {}  # 파일별 색상 저장
        self.color_index = 0  # 색상 인덱스

        # 상단 버튼 프레임
        self.button_frame = tk.Frame(self)
        self.button_frame.pack(pady=10)

        self.load_button = tk.Button(self.button_frame, text="Load CSV", command=self.load_csv)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.plot_button = tk.Button(self.button_frame, text="Plot Data", command=self.plot_selected_data)
        self.plot_button.pack(side=tk.LEFT, padx=5)

        self.zoomout_button = tk.Button(self.button_frame, text="Zoom Out", command=self.zoom_out)
        self.zoomout_button.pack(side=tk.LEFT, padx=5)

        # Pass 입력 프레임 추가
        self.pass_frame = tk.Frame(self.button_frame)
        self.pass_frame.pack(side=tk.LEFT, padx=5)
        
        tk.Label(self.pass_frame, text="Pass:").pack(side=tk.LEFT)
        self.pass_entry = tk.Entry(self.pass_frame, width=10)
        self.pass_entry.pack(side=tk.LEFT, padx=5)
        
        self.pass_plot_button = tk.Button(self.pass_frame, text="Plot Pass", command=self.plot_pass_data)
        self.pass_plot_button.pack(side=tk.LEFT, padx=5)

        self.coord_label = tk.Label(self, text="")
        self.coord_label.pack(pady=5)

        # 파일 선택 및 열 선택을 위한 노트북 위젯
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.X, expand=False, pady=0, ipady=0)

        # 그래프 영역
        self.figure = plt.Figure()
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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

    def get_distinct_color(self):
        # HSV 색상 공간에서 균등하게 분포된 색상 생성
        hue = self.color_index * 0.618033988749895  # 황금비를 사용하여 색상 분포
        saturation = 0.8
        value = 0.9
        rgb = hsv_to_rgb([hue, saturation, value])
        self.color_index += 1
        return rgb

    def load_csv(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if file_paths:
            for file_path in file_paths:
                base_name = file_path.split('/')[-1]
                file_name = base_name
                # 파일명 중복 방지
                count = 1
                while file_name in self.dataframes:
                    file_name = f"{base_name}({count})"
                    count += 1
                try:
                    df = pd.read_csv(file_path)
                except Exception as e:
                    self.coord_label.config(text=f"{file_name} 파일 읽기 오류: {e}")
                    continue
                if df.empty:
                    self.coord_label.config(text=f"{file_name} 데이터가 비어있음.")
                    continue
                self.dataframes[file_name] = df
                self.dataframes[file_name].fillna(method='ffill', inplace=True)
                self.file_colors[file_name] = self.get_distinct_color()  # 새로운 색상 할당 방식 사용
                self.create_file_tab(file_name)

    def create_file_tab(self, file_name):
        # 새로운 탭 생성
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=file_name)

        # 체크박스 프레임 (패딩 최소화)
        check_frame = tk.Frame(tab)
        check_frame.pack(pady=1, padx=1, anchor=tk.NW, fill=tk.X, expand=False)

        # 체크박스 변수 저장
        if not hasattr(self, 'check_vars'):
            self.check_vars = {}
        self.check_vars[file_name] = {}

        # 파일별 alpha(투명도) 슬라이더 추가
        if not hasattr(self, 'file_alphas'):
            self.file_alphas = {}
        alpha_frame = tk.Frame(tab)
        alpha_frame.pack(pady=0, padx=1, anchor=tk.NW, fill=tk.X, expand=False)
        tk.Label(alpha_frame, text="투명도:", font=("Arial", 9)).pack(side=tk.LEFT)
        alpha_var = tk.DoubleVar(value=0.6)
        self.file_alphas[file_name] = alpha_var
        alpha_slider = tk.Scale(alpha_frame, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=100, variable=alpha_var, showvalue=True, command=lambda e, fn=file_name: self.plot_selected_data())
        alpha_slider.pack(side=tk.LEFT)

        # 열 선택 체크박스 생성
        columns_to_plot = self.dataframes[file_name].columns[1:]
        
        # 8열로 배치하기 위한 계산
        num_columns = 8
        num_rows = (len(columns_to_plot) + num_columns - 1) // num_columns

        for i, column in enumerate(columns_to_plot):
            var = tk.BooleanVar()
            self.check_vars[file_name][column] = var
            check_button = tk.Checkbutton(check_frame, text=column, variable=var, font=("Arial", 9))
            row = i // num_columns
            col = i % num_columns
            check_button.grid(row=row, column=col, sticky=tk.W, padx=2, pady=1)

    def downsample(self, data, factor=10):
        return data[::factor]

    def plot_selected_data(self):
        if not self.dataframes:
            self.coord_label.config(text="데이터가 없습니다.")
            return

        self.figure.clear()
        self.ax_list = []
        self.ax1 = self.figure.add_subplot(111)
        self.ax_list.append(self.ax1)

        plotted = False
        for file_name, df in self.dataframes.items():
            if file_name not in self.check_vars:
                continue
            selected_columns = [col for col, var in self.check_vars[file_name].items() if var.get()]
            if selected_columns:
                alpha = self.file_alphas[file_name].get() if hasattr(self, 'file_alphas') and file_name in self.file_alphas else 0.6
                for column in selected_columns:
                    if column not in df.columns:
                        continue
                    downsampled_data = self.downsample(df[column].values)
                    downsampled_index = self.downsample(df.index.values)
                    self.ax1.plot(
                        downsampled_index,
                        downsampled_data,
                        label=f"{file_name} - {column}",
                        color=self.file_colors[file_name],
                        alpha=alpha
                    )
                    plotted = True
        self.ax1.set_xlabel('Index')
        self.ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        self.figure.tight_layout()
        self.canvas.draw()
        self.connect_zoom()
        if not plotted:
            self.coord_label.config(text="선택된 열이 없습니다. 각 파일별로 열을 선택하세요.")
        else:
            self.coord_label.config(text="")

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
        if self.dataframes:
            for ax in self.ax_list:
                ax.set_xlim(self.dataframes[list(self.dataframes.keys())[0]].index.min(),
                            self.dataframes[list(self.dataframes.keys())[0]].index.max())
                ax.set_ylim(self.dataframes[list(self.dataframes.keys())[0]].min().min(),
                            self.dataframes[list(self.dataframes.keys())[0]].max().max())
            self.canvas.draw()

    def plot_pass_data(self):
        if not self.dataframes:
            self.coord_label.config(text="데이터가 없습니다.")
            return

        try:
            target_pass = int(self.pass_entry.get())
        except ValueError:
            self.coord_label.config(text="유효한 Pass 번호를 입력하세요.")
            return

        self.figure.clear()
        self.ax_list = []
        self.ax1 = self.figure.add_subplot(111)
        self.ax_list.append(self.ax1)

        plotted = False
        for file_name, df in self.dataframes.items():
            if file_name not in self.check_vars:
                continue
            selected_columns = [col for col, var in self.check_vars[file_name].items() if var.get()]
            if selected_columns:
                alpha = self.file_alphas[file_name].get() if hasattr(self, 'file_alphas') and file_name in self.file_alphas else 0.6
                if 'Pass' not in df.columns:
                    continue
                pass_data = df[df['Pass'] == target_pass]
                if pass_data.empty:
                    continue
                pass_data = pass_data.reset_index(drop=True)
                for column in selected_columns:
                    if column != 'Pass' and column in pass_data.columns:
                        self.ax1.plot(
                            pass_data.index,
                            pass_data[column],
                            label=f"{file_name} - {column}",
                            color=self.file_colors[file_name],
                            alpha=alpha
                        )
                        plotted = True
        self.ax1.set_xlabel('Index (0부터 시작)')
        self.ax1.set_title(f'Pass {target_pass} 데이터')
        self.ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        self.figure.tight_layout()
        self.canvas.draw()
        self.connect_zoom()
        if not plotted:
            self.coord_label.config(text="해당 Pass에 데이터가 없거나, 열이 선택되지 않았습니다.")
        else:
            self.coord_label.config(text="")


if __name__ == "__main__":
    app = DataVisualizer()
    app.mainloop()
