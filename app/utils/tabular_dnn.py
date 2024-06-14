import torch.nn as nn
import torch.nn.functional as F

class DenseNN(nn.Module):

    def __init__(self,
                 input_size):
        super(DenseNN, self).__init__()

        self.dropout = nn.Dropout(0.3)
        self.dense_1 = nn.Linear(input_size, 128)
        self.dense_2 = nn.Linear(128, 2)
        self.num_labels = 2

    def forward(self,
                x,
                labels=None):

        output = F.relu(self.dense_1(x.float()))
        output = self.dense_2(output)
        logits = self.dropout(output)

        loss = None

        # Training on binary
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss_fct(logits, labels.long())

        return (loss, logits) if labels is not None else (None, logits)
