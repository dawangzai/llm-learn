from transformers import BertModel, BertConfig, BertTokenizer
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# pretrained_model = BertModel.from_pretrained(r'/mnt/workspace/model/google-bert/bert-base-chinese').to(DEVICE)
# print(pretrained_model)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(r'/mnt/workspace/model/google-bert/bert-base-chinese').to(DEVICE)
        self.fc = torch.nn.Linear(768, 2)
        # self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooler_output = self.fc(outputs.last_hidden_state[:, 0])
        return pooler_output