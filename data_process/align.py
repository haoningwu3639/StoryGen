import os
import torch
import clip
import re
import string
from PIL import Image
from deepmultilingualpunctuation import PunctuationModel
import easyocr
import numpy as np
from numpy import array, zeros, argmin, inf, equal, ndim

device = "cuda" if torch.cuda.is_available() else "cpu"
Punctmodel = PunctuationModel()
model, preprocess = clip.load("./ViT-B-16.pt", device=device)
reader = easyocr.Reader(['en'])

image_path = 'image/'
txt_path = 'img2txt/'

fns = lambda s: sum(((s,int(n))for s,n in re.findall('(\D+)(\d+)','a%s0'%s)),())

def cosine_similarity(a, b):
    a = a.reshape(a.shape[1])
    b = b.reshape(b.shape[1])
    return 1-(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def time_dist(a, b):
    a_time = fns(a)[1] * 3600 +fns(a)[3]* 60 + fns(a)[5] + fns(a)[7]*0.0001
    b_time = fns(b)[1] * 3600 +fns(b)[3]* 60 + fns(b)[5] + fns(b)[7]*0.0001
    return abs(a_time-b_time)

def align_dtw(sub_dir):
    whole_text=''
    text_timestamp = []
    text = []

    try:
        for line in open(image_path + sub_dir +'/'+ sub_dir + '.txt').readlines():
            if line !='' and line != ' ' and line != '\n':
                ListFromLine=line.strip().split('->')
                if ListFromLine[1] !='[Applause]' and ListFromLine[1] !='[Music]' and ListFromLine[1] !='[Music][Applause]' and ListFromLine[1] !='[Applause][Music]':
                    text_timestamp.append(ListFromLine[0].replace(':', '-').replace('.', '-'))
                    text.append(ListFromLine[1])
                    whole_text += ' ' + ListFromLine[1]
    except:
        print("File failure occurs.")
        return

    try:
        whole_text = Punctmodel.restore_punctuation(whole_text)
    except:
        print("Punctuation failure occurs.")
        return

    sentences = re.split(r'[.!?]+', whole_text)
    sepword = re.findall(r'[.!?]+', whole_text)
    sentences = [ x+y for x,y in zip(sentences, sepword) ] 
    sentences = [sentence for sentence in sentences if sentence != "\n" and sentence !="" and sentence !=" "]

    sen_timestamp = []
    img_timestamp = []

    l = len(text)
    tmp_i = 0

    for sentence in sentences:
        sentence = sentence.strip()
        tmp_sentence = sentence
        tmp_sentence = tmp_sentence.translate(str.maketrans("", "", string.punctuation+ " ")).lower()
        flag = 0
        for i in range(l):
            if i < tmp_i:
                continue
            if text[i].strip().translate(str.maketrans("", "", string.punctuation+ " ")).lower() in tmp_sentence or tmp_sentence in text[i].strip().translate(str.maketrans("", "", string.punctuation+ " ")).lower() or (i>0 and tmp_sentence in (text[i-1]+text[i]).strip().translate(str.maketrans("", "", string.punctuation+ " ")).lower()):
                sen_timestamp.append(text_timestamp[i])
                tmp_i = i
                flag=1
                break
        if flag !=1:
            sen_timestamp.append(text_timestamp[min(tmp_i+1,l-1)])
            tmp_i = tmp_i + 1
       
    images_path = image_path + sub_dir

    image_features = []
    sentence_features = []
    images = []

    for i in range(len(sentences)):
        sentence= sentences[i].strip()
        if len(sentence) > 77:
            sentence = sentence[:77]
        comp_sentence = clip.tokenize([sentence]).to(device)
        comp_feature = model.encode_text(comp_sentence).to('cpu').numpy()
        sentence_features.append(comp_feature)

    for image_name in sorted(os.listdir(images_path),key=fns):
        i_path = images_path + "/" + image_name
        if os.path.splitext(os.path.basename(i_path))[1] == '.jpg':        
            # ocr    
            results = reader.readtext(i_path)
            result=''
            if  len(results)!= 0: # if len=0, use vision features instead
                for i in range(len(results)):
                    result+=results[i][1]
                # text feature
                result = result.strip()
                if len(result) > 77:
                    result = result[:77]
                result = clip.tokenize([result]).to(device)
                feature = model.encode_text(result)            
            else:
                # image feature
                image = preprocess(Image.open(i_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    feature = model.encode_image(image)
            image_features.append(feature.to('cpu').numpy())
            images.append(image_name)

            image_timestamp = os.path.splitext(os.path.basename(i_path))[0]
            image_timestamp = image_timestamp.split("_")[-1]
            img_timestamp.append(image_timestamp)

    image_features = np.array(image_features)
    sentence_features = np.array(sentence_features)

    try:
        # Compute alignment relation
        r, c = len(image_features), len(sentence_features)
        D0 = zeros((r+1,c+1))
        D0[0,1:] = inf
        D0[1:,0] = inf
        D1 = D0[1:,1:]

        for i in range(r):
            for j in range(c):
                if time_dist(img_timestamp[i],sen_timestamp[j]) <= 60:
                    D1[i,j] = cosine_similarity(image_features[i],sentence_features[j])  
                else:
                    D1[i,j] = time_dist(img_timestamp[i],sen_timestamp[j])//60 + cosine_similarity(image_features[i],sentence_features[j])  

        M = D1.copy()
        for i in range(r):
            for j in range(c):
                D1[i,j] += min(D0[i,j],D0[i,j+1],D0[i+1,j])

        i,j = array(D0.shape) - 2
        p,q = [i],[j]
        while(i>0 or j>0):
            tb = argmin((D0[i,j],D0[i,j+1],D0[i+1,j]))
            if tb==0 :
                i-=1
                j-=1
            elif tb==1 :
                i-=1
            else:
                j-=1
            p.insert(0,i)
            q.insert(0,j)

        # Write to txt file
        path = list(zip(p,q))
        txt_name =  txt_path + sub_dir +".txt"
        
        print("Write to "+txt_name)
        with open(txt_name, 'w') as f:
            for i in range(len(path)):
                if i == 0:
                    f.write(images[path[i][0]])
                    f.write("->")
                if i != 0 and path[i][0] != path[i-1][0]:
                    f.write("\n")
                    f.write(images[path[i][0]])
                    f.write("->")
                f.write(sentences[path[i][1]])
                f.write('.')
    except:
        print("FastDTW failure occurs.")
                

if __name__ == '__main__':

    folders_dir = sorted(os.listdir(image_path))
    for i, sub_dir in enumerate(folders_dir):
        try:
            with torch.no_grad():
                align_dtw(sub_dir)
        except:
            print("Failure occurs at "+ sub_dir)
            continue

# CUDA_VISIBLE_DEVICES=1 python align.py