import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from model import Model
from data import MyDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('DEVICE:', DEVICE)

tokenizer = BertTokenizer.from_pretrained(r'/mnt/workspace/model/google-bert/bert-base-chinese')
def collate_fn(data):
    sentences = [item[0] for item in data]
    labels = [item[1] for item in data]
    encoding = tokenizer.batch_encode_plus(
        sentences, 
        truncation=True, 
        padding=True,
        max_length=512,
        return_tensors='pt')
    labels = torch.LongTensor(labels)
    return encoding['input_ids'], encoding['attention_mask'], encoding['token_type_ids'], labels

train_loader = DataLoader(
    dataset=MyDataset("train"), 
    batch_size=500, 
    shuffle=True, 
    drop_last=True,
    collate_fn=collate_fn
)

valid_loader = DataLoader(
    dataset=MyDataset("valid"), 
    batch_size=500, 
    shuffle=True, 
    drop_last=True,
    collate_fn=collate_fn
)

EPOCHS = 5
def start_train():
    # 定义模型
    model = Model().to(DEVICE)
    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters())
    # 定义损失函数
    loss_fn = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(EPOCHS):
        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, label = batch
            input_ids, attention_mask, token_type_ids, label = input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(DEVICE), label.to(DEVICE)
            model_output = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(model_output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每隔5个批次输出一次损失值
            if step % 5 == 0:
                acc = (model_output.argmax(dim=1) == label).sum().item() / len(label)
                print(f'Epoch: {epoch}, Step: {step}, Loss: {loss.item()}, acc: {acc}')
        # 训练完一轮通过验证集验证模型效果
        model.eval()
        val_acc = 0.0
        val_loss = 0.0
        with torch.no_grad():
            for step, batch in enumerate(valid_loader):
                input_ids, attention_mask, token_type_ids, label = batch
                input_ids, attention_mask, token_type_ids, label = input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(DEVICE), label.to(DEVICE)
                model_output = model(input_ids, attention_mask, token_type_ids)
                val_loss += loss_fn(model_output, label)
                val_acc += (model_output.argmax(dim=1) == label).sum().item()
            val_acc /= len(valid_loader)
            val_loss /= len(valid_loader)
            print(f'验证集: Epoch:{epoch}, Loss:{val_loss}, acc:{val_acc}')
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), r'/mnt/workspace/project/llm-learn/code/bert_demo/params/best_bert.pth')
                print(f'保存模型 epoch:{epoch}, val_acc:{val_acc}') 
    if epoch == EPOCHS-1:
        torch.save(model.state_dict(), r'/mnt/workspace/project/llm-learn/code/bert_demo/params/last_bert.pth')
        print(f'保存模型最后一轮 epoch:{epoch}')

if __name__ == '__main__':
    start_train()
    

