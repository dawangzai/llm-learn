from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_type):
        with open(f'project/llm-learn/code/bert_demo/data/{data_type}.csv',encoding="utf-8") as f:
            lines = f.readlines()
        self.lines = [item.strip() for item in lines]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        text = self.lines[index][:-2]
        label = (int)(self.lines[index][-1])
        return text, label

if __name__ == '__main__':
    dataset = MyDataset('train')
    print(len(dataset))
    for index, data in enumerate(dataset):
        if index < 3:
            text, label = data
            print(f'{text}:{label}')