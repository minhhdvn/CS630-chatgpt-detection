from datasets import load_dataset
import random, os, json

tmp = load_dataset('Hello-SimpleAI/HC3', 'all')['train']

data = []

did = 0
for d in tmp:
    question = d['question']
    human_answers = d['human_answers']
    gpt_answers = d['chatgpt_answers']

    for ha in human_answers:
        data.append({
            'id': did,
            'question': question,
            'answer': ha,
            'label': 'Human'
        })
        did += 1

    for ga in gpt_answers:
        data.append({
            'id': did,
            'question': question,
            'answer': ga,
            'label': 'ChatGPT'
        })
        did += 1

size = len(data)

random.seed(2023)

for _ in range(10):
    random.shuffle(data)

train_data = data[:int(0.8 * size)]
dev_data = data[int(0.8 * size): int(0.9 * size)]
test_data = data[int(0.9 * size):]

os.system('mkdir -p datasets/hc3')

with open('datasets/hc3/train.json', 'w') as f:
    json.dump(train_data, f, ensure_ascii=False)

with open('datasets/hc3/dev.json', 'w') as f:
    json.dump(dev_data, f, ensure_ascii=False)

with open('datasets/hc3/test.json', 'w') as f:
    json.dump(test_data, f, ensure_ascii=False)

print('Preprocessed: train: {}, dev: {}, test: {}'.format(len(train_data), len(dev_data), len(test_data)))
