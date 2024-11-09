import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from model import RNN_AE

# 하이퍼파라미터 설정
input_dim = 8      # 입력 데이터의 차원
hidden_dim = 16    # 은닉 상태의 차원
latent_dim = 4     # 잠재 벡터의 차원
num_layers = 1     # RNN 레이어 수
batch_size = 32
seq_len = 64
num_epochs = 10000    # 에포크 수
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# 모델 생성
model = RNN_AE(input_dim, hidden_dim, latent_dim, num_layers).to(device)
model.train()  # 학습 모드로 설정

# 손실 함수와 옵티마이저 정의
criterion = nn.MSELoss()   # 오토인코더의 경우 입력과 출력 간의 MSE 사용
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델 정보 출력
x = torch.randn(batch_size, seq_len, input_dim).to(device)  # 예시 입력 데이터
# summary(model, input_size=(batch_size, seq_len, input_dim))

# 데이터셋 생성 (예시 데이터)
# 실제 학습 데이터가 있는 경우, DataLoader로 변경하세요.
train_data = torch.randn(1000, seq_len, input_dim, dtype=torch.float32)  # 예시 데이터셋 (1000개의 샘플)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 학습 루프
for epoch in range(num_epochs):
    epoch_loss = 0.0  # 에포크별 손실 누적

    for batch in train_loader:
        optimizer.zero_grad()  # 옵티마이저 초기화

        # 모델에 입력 데이터 전달하여 순전파
        output, latent = model(batch)

        # 손실 계산 (입력과 출력 간의 차이)
        loss = criterion(output, batch)
        epoch_loss += loss.item()  # 에포크 손실에 누적

        # 역전파 및 가중치 업데이트
        loss.backward()
        optimizer.step()

    # 에포크별 평균 손실 출력
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
