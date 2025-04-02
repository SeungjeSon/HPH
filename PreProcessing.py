import pandas as pd
import os

# CSV 파일 읽기m
try:
    df = pd.read_csv('./MHPH05/Data/raw/250328/250328HM205.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('./MHPH05/Data/raw/250328/250328HM205.csv', encoding='cp949')  # 한글 파일의 경우 cp949로 시도


# 빈칸을 바로 위 행의 값으로 채웁니다.
df.fillna(method='ffill', inplace=True)

# 3. 가공된 데이터를 1/10의 크기로 줄이기
# df_reduced = df.iloc[::10, :]

# 저장 경로 설정
save_path = './MHPH05/Data/raw/250328/250328HM205_FULL.csv'
save_dir = os.path.dirname(save_path)

# 폴더가 없다면 생성
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# CSV로 저장
# df_reduced.to_csv(save_path, index=False)
df.to_csv(save_path, index=False)

print(f'파일이 저장되었습니다: {save_path}')
