import torch
import torch.nn as nn

class RNN_AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super(RNN_AE, self).__init__()

        self.encoder = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.hidden2lat = nn.Linear(hidden_dim, latent_dim)

        self.lat2hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.RNN(hidden_dim, input_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):

        _, h_n = self.encoder(x)
        latent = self.hidden2lat(h_n[-1])

        hidden = self.lat2hidden(latent).unsqueeze(0)
        output, _ = self.decoder(hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2))

        return output, latent


class CNN_AE(nn.Module):
    def __init__(self, img_size: tuple[int, int], channels=1, latent_dim=8):
        super(CNN_AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, latent_dim),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 16 * 16),
            nn.Unflatten(1, (64, 16, 16)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, 3, padding=1, output_padding=1, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, padding=1, output_padding=1, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, 3, padding=1),
        )

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output, latent


class DCCAE(nn.Module):
    def __init__(self, rnn_input_dim, img_size, rnn_hidden_dim, channels=1, latent_dim=8, rnn_layers=1):
        super(DCCAE, self).__init__()

        self.rnn_ae = RNN_AE(rnn_input_dim, rnn_hidden_dim, latent_dim, rnn_layers)
        self.cnn_ae = CNN_AE(img_size, latent_dim=latent_dim, channels=channels)

    def forward(self, rnn_x, img_x):
        rnn_output, rnn_latent = self.rnn_ae(rnn_x)
        img_output, img_latent = self.cnn_ae(img_x)

        return rnn_output, img_output, rnn_latent, img_latent


def CCA_loss(h1, h2, dim, alpha=1e-3):
    h1_mean = h1 - h1.mean(0)
    h2_mean = h2 - h2.mean(0)

    sigma_h1 = h1_mean.T @ h1_mean / (h1_mean.shape[0] - 1)
    sigma_h2 = h2_mean.T @ h2_mean / (h2_mean.shape[0] - 1)
    sigma_h1h2 = h1_mean.T @ h2_mean / (h1_mean.shape[0] - 1)

    sigma_h1_inv = torch.inverse(sigma_h1 + alpha * torch.eye(sigma_h1.shape[1]))
    sigma_h2_inv = torch.inverse(sigma_h2 + alpha * torch.eye(sigma_h2.shape[1]))

    T = sigma_h1_inv @ sigma_h1h2 @ sigma_h2_inv @ sigma_h1h2.T
    eigvals = torch.linalg.eigvals(T)

    loss = -torch.sum(torch.sqrt(eigvals[:dim]))
    return loss


if __name__ == '__main__':
    from torchinfo import summary
    from matplotlib import pyplot as plt

    # RNN_AE 모델 구조 확인
    rnn_ae = RNN_AE(input_dim=8, hidden_dim=64, latent_dim=8, num_layers=8)
    summary(rnn_ae, input_size=(8, 128, 8))

    # CNN_AE 모델 구조 확인
    cnn_ae = CNN_AE((64, 64), channels=3, latent_dim=8)
    summary(cnn_ae, input_size=(1, 3, 64, 64))

    # # MultiModal_AE 모델 구조 확인
    img_size = (64, 64)
    multi_modal_ae = DCCAE(rnn_input_dim=8, img_size=img_size, channels=3, rnn_hidden_dim=64, latent_dim=4, rnn_layers=8)
    summary(multi_modal_ae, input_data=[torch.randn(32, 64, 8), torch.randn(32, 3, 64, 64)])

    # CNN_AE 모델 테스트
    model = CNN_AE((64, 64), channels=3, latent_dim=8)
    test_input = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        reconstructed_output, _ = model(test_input)
        print(reconstructed_output.shape)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(test_input[0, 0].cpu().numpy())
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(reconstructed_output[0, 0].cpu().numpy())
    axes[1].set_title("Reconstructed Image")
    axes[1].axis('off')

    plt.show()