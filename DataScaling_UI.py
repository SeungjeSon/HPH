import pandas as pd
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class DataScalingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("데이터 스케일링 도구")
        self.root.geometry("600x400")
        
        # 스타일 설정
        style = ttk.Style()
        style.configure("TButton", padding=5)
        style.configure("TLabel", padding=5)
        
        # 메인 프레임
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 파일 선택
        self.file_path = tk.StringVar()
        ttk.Label(main_frame, text="CSV 파일:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.file_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(main_frame, text="파일 선택", command=self.select_file).grid(row=0, column=2)
        
        # 다운스케일링 값
        ttk.Label(main_frame, text="다운스케일링 값:").grid(row=1, column=0, sticky=tk.W, pady=10)
        self.scale_value = tk.StringVar(value="10")
        scale_spin = ttk.Spinbox(main_frame, from_=1, to=100, textvariable=self.scale_value, width=10)
        scale_spin.grid(row=1, column=1, sticky=tk.W, pady=10)
        
        # 처리 버튼
        ttk.Button(main_frame, text="처리 시작", command=self.process_data).grid(row=2, column=0, columnspan=3, pady=20)
        
        # 상태 표시
        self.status_var = tk.StringVar(value="준비됨")
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=3, column=0, columnspan=3)
        
    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="CSV 파일 선택",
            filetypes=[("CSV 파일", "*.csv"), ("모든 파일", "*.*")]
        )
        if file_path:
            self.file_path.set(file_path)
            self.status_var.set("파일이 선택되었습니다.")
    
    def process_data(self):
        file_path = self.file_path.get()
        if not file_path:
            messagebox.showwarning("경고", "파일을 선택해주세요.")
            return
            
        try:
            self.status_var.set("파일 처리 중...")
            self.root.update()
            
            # CSV 파일 읽기
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='cp949')
            
            # 빈칸을 바로 위 행의 값으로 채우기
            df.fillna(method='ffill', inplace=True)
            
            # 다운스케일링
            scale_value = int(self.scale_value.get())
            df_reduced = df.iloc[::scale_value, :]
            
            # 저장 경로 설정
            file_name = os.path.basename(file_path)
            file_name_without_ext = os.path.splitext(file_name)[0]
            save_path = os.path.join(
                os.path.dirname(file_path),
                f"{file_name_without_ext}_downscaling_{scale_value}.csv"
            )
            
            # 저장 디렉토리 확인 및 생성
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # CSV로 저장 (한글 인코딩 문제 해결)
            df_reduced.to_csv(save_path, index=False, encoding='utf-8-sig')
            
            self.status_var.set(f"처리 완료: {save_path}")
            messagebox.showinfo("완료", f"파일이 저장되었습니다:\n{save_path}")
            
        except Exception as e:
            self.status_var.set("오류 발생")
            messagebox.showerror("오류", f"처리 중 오류가 발생했습니다:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DataScalingApp(root)
    root.mainloop()
