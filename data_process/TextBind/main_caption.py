import os
import re
import torch
from PIL import Image
from tqdm import tqdm

from data_utils import preprocess_mim
from utils import parse_args, build_model_and_processor

def postprocesss(s, args):

    s = s.replace("<image>" * args.num_query_tokens, "<image>")
    s = s.replace(" ".join(["<image>"] * args.num_query_tokens), "<image>")
    s = s.replace("  ", " ")
    pattern = "<start>.*?<end>"
    s = re.sub(pattern, "<image>", s)
    return s

def inference(
    data, 
    args,
    model,
    tokenizer,
    image_processor,
    device,
    ):
    
    # process mim input
    inputs = preprocess_mim(data, tokenizer, image_processor, args, False)        
    
    input_ids = torch.LongTensor(inputs["input_ids"]).unsqueeze(0).to(device)
    input_images = (None if inputs["input_images"] is None else torch.stack(inputs["input_images"], 0).unsqueeze(0).to(device))
    input_image_index = (None if inputs["input_image_index"] is None else torch.LongTensor(inputs["input_image_index"]).unsqueeze(0).to(device)) 
    
    outputs = model.cache_generation(
        input_ids=input_ids,
        input_images=input_images,
        input_image_index=input_image_index,
        caption_start_id=args.caption_start_id,
        caption_end_id=args.caption_end_id,
        tokenizer=tokenizer,
        max_output_length=args.max_output_length,
        top_p=args.top_p,
    )
    
    total_text = tokenizer.decode(outputs["sequences"][0].tolist(), skip_special_tokens=False)
    generation = total_text.split("[/INST]")[-1].strip()
     
    data["conversation"].append(
        {
            "role": "assistant",
            "content": generation,
            "caption_list": outputs["caption_list"],
        }
    )
    
    for turn in data["conversation"]:
        turn['content'] = postprocesss(turn['content'], args)
        
    return data

class MIMPipeline:
    def __init__(self, args, device):

        model, tokenizer, image_processor = build_model_and_processor(args)
        model = model.half().to(device) if args.fp16 else model.to(device)
        model.eval()

        self.args = args
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
    
    def run(self, data):
        data = inference(data, self.args, self.model, self.tokenizer, self.image_processor, self.device)
        
        return data

def evaluate(args, device):

    image_path = '../../StorySalon/Image/StoryWeaver/'
    text_path = '../../StorySalon/Text/Narrative_Filtered/StoryWeaver/'
    caption_path = '../../StorySalon/Text/Caption/StoryWeaver/'
    
    agent = MIMPipeline(args, device)
    commend = "<image> Provide a short and precise English description of the image displayed in 50 words. The corresponding story narrative is: '"
    
    folders = sorted(os.listdir(os.path.join(image_path)))
    
    # folders = folders[2800:]
    for i, folder in enumerate(folders):
        caption_folder = os.path.join(caption_path, folder)
        text_folder = os.path.join(text_path, folder)
        
        if not os.path.exists(caption_folder):
            os.mkdir(caption_folder)
        
        images = sorted(os.listdir(os.path.join(image_path, folder)))
        for image in tqdm(images):
            temp_path = os.path.join(image_path, folder, image)
            print(temp_path)
            img = Image.open(temp_path).convert("RGB")

            text_save_path = os.path.join(text_folder, image[:-4] + '.txt')
            caption_save_path = os.path.join(caption_folder, image[:-4] + '.txt')
            
            if not os.path.exists(caption_save_path):

                with open(text_save_path, 'r') as f:
                    text = f.read()
                    
                content = commend + text
                data = {
                    "conversation": [
                        {
                            "role": "system",
                            "content": "You are an image captioning robot, you can caption an image with the corresponding narrative. Don't repeat the narrative, and all the responses must be English, summarized to less than 50 words.",
                            "image_list": [],
                        },
                        {
                            "role": "user",
                            "content": content,
                            "image_list": [img],
                        },
                    ]
                }
                
                result = agent.run(data)
                # print(result)
                
                if len(result['conversation'][-1]['caption_list']) > 0:                
                    caption = result['conversation'][-1]['caption_list'][0] # caption list
                    
                else:
                    summarize_command = "Please summarize the following sentetnces into less than 50 words, must be in English. The sentence is: "
                    summarize_content = result['conversation'][-1]['content']
                    
                    summarize_data = {
                        "conversation": [
                            {
                                "role": "user",
                                "content": summarize_command + summarize_content,
                                "image_list": [],
                            },
                        ]
                    }
                    summarize_result = agent.run(summarize_data)
                    # print(summarize_result)
                    caption = summarize_result['conversation'][-1]['content']
                    pattern = r':\n(.*?)</s>'
                    matches = re.search(pattern, caption)
                    if not matches:
                        pattern = r':(.*?)</s>'
                        matches = re.search(pattern, caption)
                    if matches:
                        caption = matches.group(1).strip()
                    # caption = result['conversation'][-1]['content']
                print(caption)
                with open(caption_save_path, 'w') as f:
                    f.write(caption)

            else:
                print("skip " + caption_save_path)

if __name__ == "__main__":

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(args, device)
