#!/usr/bin/env python

# Copyright (c) Xiaochen Zheng.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import argparse
import numpy as np
import json, csv
from PIL import Image
from io import BytesIO
import base64
import pandas as pd
import pydicom

from tqdm import tqdm

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def parse_agrs():

    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--save_dir', type=str, default='C:/Users/16377/Downloads/mimic-cxr/2.0.0', help='the path to the directory containing the data.')
    parser.add_argument('--image_dir', type=str, default="C:/Users/16377/Downloads/mimic-cxr/2.0.0", help='the path to the directory containing the data.')
    # parser.add_argument('--ann_path', type=str, default='here is the annotation file that you need to generate', help='the path to the directory containing the data.')
    parser.add_argument('--chest_imagenome_dir', type=str, default="C:/Users/16377/Downloads/chest-imagenome-dataset-1.0.0/chest-imagenome-dataset-1.0.0", help='the path to the directory containing the data.')
    args = parser.parse_args()

    ###

    return args

def generate_silver_captions(scene_graph_dir, save_dir):
    """
    Generate phrases from scene graph annotations.
    """
    csv_file = os.path.join(save_dir, "silver_captions.csv")
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["patient_id", "study_id", "image_id", "object_id", "caption", "x1", "y1", "x2", "y2", "width", "height"])

        json_files = [f for f in os.listdir(scene_graph_dir) if f.endswith(".json")]
        for json_file in json_files:
            # json_file: 7ed2af88-0a6e3b4b-92d6feea-c649176c-aea8a910_SceneGraph.json
            # take the part before_, which is the image_id
            image_id = json_file.split("_")[0]
            json_path = os.path.join(scene_graph_dir, json_file)

            # phrases example in each json file:
            #{
            # ...
            # "attributes": [
            #     {
            #      ....
            #      "phrases": [
            #         "No pleural effusion, pneumothorax or pulmonary edema."
            #          ],
            #       ....
            #      "object_id": "....."
            #     },
            #     {
            #      ....
            #      "phrases": [
            #          "Heart\n size is mildly enlarged with normal mediastinal contour and hila.",
            #          "No pleural effusion, pneumothorax or pulmonary edema."
            #       ],
            #      ....
            #      "object_id": "......"
            #     },
            #     ....

            with open(json_path) as f:
                data = json.load(f)
                patient_id = data.get("patient_id", "")
                study_id = data.get("study_id", "")
                objects = data.get("objects", [])
                attributes = data.get("attributes", [])

                for obj in objects:
                    object_id = obj.get("object_id", "")
                    x1, y1, x2, y2 = obj.get("x1", ""), obj.get("y1", ""), obj.get("x2", ""), obj.get("y2", "")
                    width, height = obj.get("width", ""), obj.get("height", "")

                    matching_phrases = []
                    for attribute in attributes:
                        if attribute.get("object_id", "") == object_id:
                            matching_phrases.extend(attribute.get("phrases", []))
                    caption = " ".join([phrase.strip() for phrase in matching_phrases])
            
                    writer.writerow([patient_id, study_id, image_id, object_id, caption, x1, y1, x2, y2, width, height])

    print(f"Silver phrases saved to: {csv_file}")


#def transform_image_to_base64(file_name):
def transform_dicom_to_base64(file_name):
    try:
        # img = Image.open(file_name) # path to file
        dcm = pydicom.dcmread(file_name)
        print(f"Successfully loaded dcm: {file_name}")
        pxiel_array_numpy = dcm.pixel_array
        img = Image.fromarray(pxiel_array_numpy)
        img_buffer = BytesIO()
        img_save = img.save(img_buffer, format="PNG")
        byte_data = img_buffer.getvalue()
        base64_str = base64.b64encode(byte_data) # bytes
        base64_str = base64_str.decode("utf-8") # str
        return base64_str
    except Exception as e:
        print(f"Error loading image: {file_name}")
        print(f"Exception: {str(e)}")
        return None

def main():
    # parse arguments
    args = parse_agrs()

    scene_graph_dir = os.path.join(args.chest_imagenome_dir, "silver_dataset/scene_graph/scene_graph")

    generate_silver_captions(scene_graph_dir, args.save_dir)

    print(f"* * * * * * * * * * * *{bcolors.HEADER} Start generating {bcolors.ENDC}* * * * * * * * * * * *")
    print(f"Task: {bcolors.OKCYAN}VL-visual grounding{bcolors.ENDC}, Dataset: {bcolors.BOLD}MS-CXR{bcolors.ENDC}")
    print('|')
    print('V')

    # Load silver captions from the generated CSV file
    silver_captions_file = os.path.join(args.save_dir, "silver_captions.csv")
    ann = pd.read_csv(silver_captions_file)

    # get the train
    train = pd.read_csv(args.chest_imagenome_dir+'/silver_dataset/splits/train.csv')
    dicom_id = train['dicom_id'].tolist()
    # train_split = json.loads(open("C:/Users/16377/Downloads/MIMIC_CXRs/train.csv", 'r').read())['train']
    
    data=[]
    nsamples = 0
    for idx, row in tqdm(train.iterrows(), total=train.shape[0]):
        dicom_id = row['dicom_id']
        dicom_path = os.path.join(args.image_dir, row['path'])
        if not os.path.exists(dicom_path):
            # print (f"Image path {dicom_path} does not exist in image_dir {args.image_dir}")
            continue

        matching_captions = ann[ann['image_id'] == dicom_id]
        if matching_captions.shape[0] == 0:
            # print(f"No captions found for image id {dicom_id}")
            continue
        
        for _, caption_row in matching_captions.iterrows():
            caption = caption_row['caption']
            x1 = caption_row['x1']
            y1 = caption_row['y1']
            x2 = caption_row['x2']
            y2 = caption_row['y2']
            base64_str = transform_dicom_to_base64(dicom_path)
            if base64_str is None:
                print(f"Error converting DICOM to base64: {dicom_path}")
                continue
            print(f"base64_str: {base64_str}")
        # uniq-id,
        # image (base64 string),
        # caption,
        # question,
        # answer,
        # ground-truth objects (objects appearing in the caption or question),
        # dataset name (source of the data)
        # task type (caption, qa or visual_grounding)
        # 'uniq-id': dataset(mimimc:1, vindr:2, ms-cxr:3), meta-task(vl:1, v:2, l:3), subtask(...), order(0,1,2,3)
        # e.g. ms-cxr with visual_grounding, 1st: 1130*10000+idx = 11300000
            x= {
                'uniq-id':3130*10000+idx,
                'base64':base64_str,
                'caption':caption,
                'question':'',
                'answer':f"{x1},{y1},{x2},{y2}",
                'objects':'',
                'dataset_name':'ms-cxr',
                'task_type':'visual_grounding',
            }
            data.append(x)
            nsamples += 1
    print(f'In total, we collected {nsamples} samples.')


    with open(os.path.join(args.save_dir, 'vision_language_examples.tsv'), 'at') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        # Write column headers
        tsv_writer.writerow([
            "uniq-id",
            "base64",
            "caption",
            "question",
            "answer",
            "objects",
            "dataset_name",
            "task_type"
        ])
        for example in tqdm(data):
            tsv_writer.writerow([
                example['uniq-id'],
                example['base64'],
                example['caption'],
                example['question'],
                example['answer'],
                example['objects'],
                example['dataset_name'],
                example['task_type']
            ])

    # with open(os.path.join(args.save_dir, 'negative_sample', 'all_captions.txt'), 'at') as txt_file:
    #     for example in tqdm(data):
    #         txt_file.write(example['caption']+"\n")  
    
    #  with open(os.path.join(args.save_dir, 'negative_sample', 'all_captions.txt'), 'r') as infile:
    #     lines = infile.readlines()

    # if lines and lines[-1] == '\n':
    #     lines = lines[:-1]
    # lines = set(lines)

    # with open(os.path.join(args.save_dir, 'negative_sample', 'all_captions.txt'), 'w') as outfile:
    #     outfile.writelines(lines)
    
    print('|')
    print('V')
    print(f"* * * * * * Finished generating visual grounding data with MS-CXR * * * * * *\n")

if __name__ == '__main__':
    main()