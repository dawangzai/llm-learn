from datasets import Dataset, load_from_disk

def prepare_data():
    datas = load_from_disk(r'/Users/wangzai/大模型/projects/llm-learn/chapter/5/L2/day04-基于BERT模型的自定义微调训练/demo_04/data/ChnSentiCorp')
    print(datas)
    train = datas['train'].to_csv(path_or_buf=r'/Users/wangzai/大模型/projects/llm-learn/code/bert_demo/data/train.csv')
    validation = datas['validation'].to_csv(path_or_buf=r'/Users/wangzai/大模型/projects/llm-learn/code/bert_demo/data/validation.csv')
    test = datas['test'].to_csv(path_or_buf=r'/Users/wangzai/大模型/projects/llm-learn/code/bert_demo/data/test.csv')




if __name__ == '__main__':
    prepare_data()