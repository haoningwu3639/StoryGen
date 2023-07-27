import os
import argparse
import openai
from PIL import Image
from chatcaptioner.chat import set_openai_key, caption_image
from chatcaptioner.blip2 import Blip2
from tqdm import tqdm

def parse():
    parser = argparse.ArgumentParser(description='Generating captions in test datasets.')
    parser.add_argument('--n_rounds', type=int, default=5, 
                        help='Number of QA rounds between GPT3 and BLIP-2. Default is 10, which costs about 2k tokens in GPT3 API.')
    parser.add_argument('--n_blip2_context', type=int, default=1, 
                        help='Number of QA rounds visible to BLIP-2. Default is 1, which means BLIP-2 only remember one previous question. -1 means BLIP-2 can see all the QA rounds')
    parser.add_argument('--model', type=str, default='chatgpt', choices=['gpt3', 'chatgpt', 'text-davinci-003', 'text-davinci-002', 'davinci', 'gpt-3.5-turbo', 'FlanT5XXL', 'OPT'],
                        help='model used to ask question. can be gpt3, chatgpt, or its concrete tags in openai system')
    parser.add_argument('--device_id', type=int, default=0, help='Which GPU to use.')

    args = parser.parse_args()
    return args

def main(args):
    # Set OpenAI
    openai_key = "sk-InputYourOpenAIKey" 
    
    set_openai_key(openai_key)
    openai.api_base = "https://api.openai-proxy.com/v1"
    # Load BLIP-2
    
    video_root = '../../StorySalon/'
    video_image_dir = os.path.join(video_root, 'image')
    
    folders = sorted(os.listdir(video_image_dir)) # 00001
    video_image_folders = [os.path.join(video_image_dir, folder) for folder in folders]
    video_image_list = []
        
    for video in video_image_folders: # video: image_folder, /StorySalon/image/00001
        images = sorted(os.listdir(video))
        for image in images:
            video_image_list.append(os.path.join(video, image))
    
    blip2s = {
        'FlanT5 XXL': Blip2('FlanT5 XXL', device_id=args.device_id, bit8=True), # load BLIP-2 FlanT5 XXL to GPU0. Too large, need 8 bit. About 20GB GPU Memory
    }
    
    if args.model == 'FlanT5XXL':
        question_model = blip2s['FlanT5 XXL']
    elif args.model == 'OPT':
        question_model = Blip2('OPT6.7B', device_id=2, bit8=True)
    else:
        question_model = args.model
    
    for blip2_tag, blip2 in blip2s.items():
        for img_path in tqdm(video_image_list):
            text_path = img_path.replace('image', 'text_caption')[:-4] + '.txt'
            img = Image.open(img_path)
    
            try:
                result = caption_image(blip2=blip2, image=img, 
                                model=question_model,
                                n_rounds=args.n_rounds, 
                                n_blip2_context=args.n_blip2_context,
                                print_mode='no'
                                )
                with open(text_path, 'w') as f:
                    f.write(result['ChatCaptioner']['caption'])
                print(result['ChatCaptioner']['caption'])
            except:
                with open('failure.txt', 'a') as g:
                    g.write(img_path)

if __name__ == '__main__':
    args = parse()
    main(args)