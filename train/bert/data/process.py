import json

if __name__ == '__main__':
    with open('./0006.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    res = []
    for line in lines:
        data = json.loads(line)
        title, desc, answer = data['title'], data['desc'], data['answer']
        title: str = title.replace('\n', '').replace('\r', '')
        desc: str = desc.replace('\n', '').replace('\r', '')
        answer: str = answer.replace('\n', '').replace('\r', '')

        s_len = 0
        while title[-s_len:] == desc[:s_len] and s_len < min(len(title), len(desc)):
            s_len += 1
        s_len -= 1

        text = title + desc[s_len:] + answer
        res.append({
            'text': text,
            'lang': 'zh-CN'
        })
    with open('./data.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False)

