import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

##################################################
# 1) 스트로크 분할 함수 (수정해서 인덱스도 반환)
##################################################
def segment_strokes_auto(pressure_array, threshold=5.0, min_stroke_len=100, min_gap=20):
    """
    압력이 threshold를 초과하면 스트로크 시작,
    threshold 이하로 min_gap 샘플 연속되면 스트로크 종료로 판단.
    """
    strokes = []
    indices = []
    in_stroke = False
    stroke_start_idx = None
    below_count = 0

    for i, p in enumerate(pressure_array):
        if not in_stroke:
            if p > threshold:
                in_stroke = True
                stroke_start_idx = i
                below_count = 0
        else:
            if p < threshold:
                below_count += 1
            else:
                below_count = 0

            # threshold 이하로 min_gap 샘플 연속되면 스트로크 종료
            if below_count >= min_gap:
                stroke_end_idx = i - min_gap
                if (stroke_end_idx - stroke_start_idx) >= min_stroke_len:
                    stroke_data = pressure_array[stroke_start_idx:stroke_end_idx]
                    strokes.append(stroke_data)
                    indices.append(stroke_start_idx)
                in_stroke = False
                stroke_start_idx = None
                below_count = 0

    return strokes, indices


##################################################
# 2) 리샘플링 함수
##################################################
def resample_stroke(stroke_array, fixed_length=200):
    """
    가변 길이 스트로크를 고정 길이로 보간 (1차 interp)
    """
    original_len = len(stroke_array)
    if original_len == 0:
        return np.zeros(fixed_length)
    original_x = np.linspace(0, 1, original_len)
    target_x = np.linspace(0, 1, fixed_length)
    resampled = np.interp(target_x, original_x, stroke_array)
    return resampled

##################################################
# 3) 로컬 Min-Max 정규화 함수
##################################################
def normalize_stroke(stroke_array):
    """
    (X - min) / (max - min)으로 0~1 스케일링
    """
    min_val = np.min(stroke_array)
    max_val = np.max(stroke_array)
    if max_val - min_val < 1e-9:
        return np.zeros_like(stroke_array)
    return (stroke_array - min_val) / (max_val - min_val)


##################################################
# 4) PyTorch Dataset (Autoencoder용)
##################################################
class StrokeDataset(Dataset):
    def __init__(self, stroke_arrays):
        # stroke_arrays: (N, fixed_length) 형태
        self.data = torch.tensor(stroke_arrays, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        # Autoencoder => (input, target) 동일
        return x, x


##################################################
# 5) Autoencoder 모델 정의
##################################################
class Autoencoder(nn.Module):
    def __init__(self, input_dim=200, latent_dim=20):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


##################################################
# 6) 학습 및 이상 점수 계산 함수
##################################################
def train_autoencoder(model, dataloader, epochs=10, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch + 1}/{epochs}] Loss: {avg_loss:.6f}")


def compute_anomaly_scores(model, dataloader):
    """
    Autoencoder 재구성 오차(MSE) -> 이상 점수
    """
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    scores = []

    with torch.no_grad():
        for batch_x, _ in dataloader:
            recon = model(batch_x)
            loss_per_sample = criterion(recon, batch_x).mean(dim=1)  # (batch,)
            scores.extend(loss_per_sample.cpu().numpy())
    return np.array(scores)


##################################################
# 7) 메인 실행부 예시 (이상 인덱스 표시 추가)
##################################################
if __name__ == "__main__":

    ########## A. 정상 데이터 ##########
    df_normal = pd.read_csv("./MHPH05/Data/raw/250306/250306_5WT_2_PM105_Data.csv", encoding="cp949")
    pressure_normal = df_normal['Pressure1(bar)'].dropna().values
    print(f"[INFO] 정상 데이터 길이: {len(pressure_normal)}")

    threshold = 50.0
    min_gap = 20
    min_stroke_len = 500

    # (1) 스트로크 분할 -> (strokes, indices)
    strokes_normal, indices_normal = segment_strokes_auto(
        pressure_normal, threshold=threshold,
        min_stroke_len=min_stroke_len, min_gap=min_gap
    )
    print(f"[INFO] 정상 스트로크 개수: {len(strokes_normal)}")

    # (2) 리샘플링 & 정규화
    fixed_length = 200
    normal_resampled_scaled = []
    for stk in strokes_normal:
        res = resample_stroke(stk, fixed_length)
        norm_res = normalize_stroke(res)
        normal_resampled_scaled.append(norm_res)

    # Dataset & DataLoader
    train_dataset = StrokeDataset(normal_resampled_scaled)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # (3) Autoencoder 학습
    model = Autoencoder(input_dim=fixed_length, latent_dim=20)
    train_autoencoder(model, train_loader, epochs=20, lr=1e-3)

    # (4) 정상 데이터 오차 -> 임계치(Mean+3σ)
    normal_scores = compute_anomaly_scores(model, train_loader)
    mean_score = np.mean(normal_scores)
    std_score = np.std(normal_scores)
    threshold_score = mean_score + 9.0 * std_score

    print(f"[INFO] 정상 데이터 MSE 평균: {mean_score:.6f}, 표준편차: {std_score:.6f}")
    print(f"[INFO] 임계값(Threshold) = {threshold_score:.6f}  (Mean+3σ)")

    ########## B. 이상 데이터 ##########
    df_abnormal = pd.read_csv("./MHPH05/Data/raw/250328/250328HM205.csv", encoding="cp949")
    pressure_abnormal = df_abnormal['Pressure1(bar)'].dropna().values
    print(f"[INFO] 이상 데이터 길이: {len(pressure_abnormal)}")

    # (1) 스트로크 분할
    strokes_abnormal, indices_abnormal = segment_strokes_auto(
        pressure_abnormal, threshold=threshold,
        min_stroke_len=min_stroke_len, min_gap=min_gap
    )
    print(f"[INFO] 이상 스트로크 개수: {len(strokes_abnormal)}")

    # (2) 리샘플링 & 정규화
    abnormal_resampled_scaled = []
    for stk in strokes_abnormal:
        res = resample_stroke(stk, fixed_length)
        norm_res = normalize_stroke(res)
        abnormal_resampled_scaled.append(norm_res)

    abnormal_dataset = StrokeDataset(abnormal_resampled_scaled)
    abnormal_loader = DataLoader(abnormal_dataset, batch_size=32, shuffle=False)

    # (3) 재구성 오차 -> 이상 판정
    abnormal_scores = compute_anomaly_scores(model, abnormal_loader)

    abnormal_count = 0
    total_ab_strokes = len(abnormal_scores)

    # (4) 이상 스트로크와 인덱스 출력
    print("\n===== 이상 스트로크 판별 결과 =====")
    for i, score in enumerate(abnormal_scores):
        if score > threshold_score:
            abnormal_count += 1
            # indices_abnormal[i]가 원본 압력 데이터에서의 시작 인덱스
            print(f" - Stroke {i} : MSE={score:.6f}, 시작인덱스={indices_abnormal[i]} (이상)")

    print(f"\n[결과] 이상 스트로크: {abnormal_count} / {total_ab_strokes}")

    # # (선택) 히스토그램 시각화
    # plt.hist(abnormal_scores, bins=30)
    # plt.axvline(x=threshold_score, color='r', linestyle='--', label='Threshold')
    # plt.title("Abnormal Data Reconstruction Error")
    # plt.xlabel("Reconstruction Error(MSE)")
    # plt.ylabel("Count")
    # plt.legend()
    # plt.show()
