import json
def combine():    
    # highlights
    with open('vscum_data/overall_highlights.txt','r') as file:
        content = file.readlines()
    content = [line.strip() for line in content if line.strip()]
    overall_highlights = [json.loads(line) for line in content]


if __name__ == ''