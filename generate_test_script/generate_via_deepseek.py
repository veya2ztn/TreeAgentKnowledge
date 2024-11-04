
from tqdm.auto import tqdm
import os,json
base_url = "https://api.gptsapi.net/v1";model    = "gpt-4o-mini" 
base_url = "https://api.deepseek.com/beta";model    = "deepseek-chat" 
with open(".openai.conf.json",'r') as f:
    openai_config = json.load(f)
api_key  = openai_config[base_url]['api_key'] #

# import os
# for var in "http_proxy https_proxy HTTP_PROXY HTTPS_PROXY".split():
#     os.environ[var]=""

# base_url = "http://10.140.24.11:30000/v1"

max_tokens= 16384
import os
os.environ['OPENAI_API_KEY']=api_key
os.environ['OPENAI_API_BASE_URL']=base_url
from tree_operations import *
import numpy as np
root = "results/single_to_single/deepseek"
logP = "results/single_to_single/deepseek/logs"
os.makedirs(root,exist_ok=True)
os.makedirs(logP,exist_ok=True)
async def process_contents(tree, contents_split):
    for round, content_list in enumerate(tqdm(contents_split)):
        already_passed_paper = set(tree.get_whole_represented_pool())
        content_list = [content for content in content_list if int(content.id_address) not in already_passed_paper]
        await tree.dispatch_multiple_contents(list(content_list))
        await tree.backpropagate(do_evolve=True)
        logs = tree.clean_after_retrieve_all_logs()
        with open(f"{logP}/deepseek.logs.{round}.json",'w') as f:
            json.dump(logs,f)
        tree.save_to_json(f"{root}/deepseek.snap.{round}.json")
if __name__ == "__main__":
    with open('extra_content.txt','r') as f:
        content = f.read()
    tree = NodeTree.load_from_file("TreeKnowledge.example.json")

    Content_root="custom_collection/graphrag/whole/summary_with_logic_position"
    contents = []
    for i,filename in enumerate(tqdm(os.listdir(Content_root))):
        filepath = os.path.join(Content_root,filename )
        with open(filepath,'r') as f:
            content = f.read()
        contents.append(Content(i,content))
    
    np.random.shuffle(contents)
    contents_split = np.array_split(contents,100)
    asyncio.run(process_contents(tree, contents_split))