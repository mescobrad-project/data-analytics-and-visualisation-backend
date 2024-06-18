import torch.nn as nn

class AutoEncoderNN(nn.Module):

    def __init__(self, input_size):
        super(AutoEncoderNN, self).__init__()

        self.input_size = input_size  # number of features

        # Encoder Layers
        self.enc_dense_1 = nn.Linear(self.input_size, 256)
        self.enc_dense_2 = nn.Linear(256, 128)
        self.enc_dense_3 = nn.Linear(128, 16)

        # Decoder Layers
        self.dec_dense_1 = nn.Linear(16, 128)
        self.dec_dense_2 = nn.Linear(128, 256)
        self.dec_dense_3 = nn.Linear(256, self.input_size)

        # Classification (2 classes)
        self.classification_layer = nn.Linear(16, 2)
        self.num_labels = 2

        # Loss
        self.alpha = 0.9

    def forward(self, x, labels=None):
        # Encoder
        output_enc_dense_1 = self.enc_dense_1(x.float())
        output_enc_dense_2 = self.enc_dense_2(output_enc_dense_1)
        latent_representation = self.enc_dense_3(output_enc_dense_2)

        # Decoder
        output_dec_dense_1 = self.dec_dense_1(latent_representation)
        output_dec_dense_2 = self.dec_dense_2(output_dec_dense_1)
        output_dec_dense_3 = self.dec_dense_3(output_dec_dense_2)

        # Classification
        logits = self.classification_layer(latent_representation)

        # Reconstruction Loss
        loss_fct = nn.MSELoss()
        reconstruction_loss = loss_fct(output_dec_dense_3, x)

        # Classification Loss
        classification_loss = 0
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(logits, labels.long())

        # Combination of losses
        total_loss = (1 - self.alpha) * classification_loss + self.alpha * reconstruction_loss

        # Final output
        return (total_loss, logits) if labels is not None else (None, logits)
