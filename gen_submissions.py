import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import tensorflow as tf
from one_shot_classification import *
from demo_one_shot_detection import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Segment Aything with siamese classification", add_help=True)

    parser.add_argument("--shelf_dir", "-i", type=str, required=True, help="path to image file")
    parser.add_argument("--query_dir", "-qi", type=str, default="", required=True, help="path to query image file")
    
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )
    parser.add_argument("--saved_model_dir", type=str, default='',
                        help="Where restore model parameters from.")
    args = parser.parse_args()

    # load arguments
    shelf_dir = args.shelf_dir
    query_dir = args.query_dir
    output_dir = args.output_dir


     ## load siamese model
    saved_model_dir = '/media/saket/92043048-cd87-4d4c-a516-022ae8564c01/Projects/ML-CV-MODEL-object_classification/SIAM-192'
    tf.keras.backend.clear_session()    

    ## allow gpu growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    gpus = ['gpu:'+gpu.name[-1] for gpu in gpus]
    print(f'GPUs : {gpus}')    

    ## instantiate the classification model
    print('Loading Model...', end='\r')
    model_siamese = tf.saved_model.load(saved_model_dir)


    ## load SAM
    sam_checkpoint = "./sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    
    

    for shelf_image_name in os.listdir(shelf_dir):
        shelf_image_path = os.path.join(shelf_dir, shelf_image_name)            
        shelf_image_class = shelf_image_name[2:-4]

        for query_image_name in os.listdir(query_dir):
            query_image_path = os.path.join(query_dir, query_image_name)
            query_image_class = query_image_name[2:-4]
    
            image = cv2.imread(shelf_image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            query_image = cv2.imread(query_image_path)
            query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)            

            masks = mask_generator.generate(image)
            crops_and_bboxes = get_crops_and_bboxes(image, masks)

            
            query_embed = get_embeddings(query_image, model_siamese)

            final_bboxes = []
            i = 0
            # print('Totoal segments =', len(crops_and_bboxes))
            for crop, bbox in crops_and_bboxes:
                if 0 in crop.shape:
                    continue
                if one_shot_match(crop, query_embed, model_siamese, from_query_embed = True):
                    # cv2.imwrite(f"./{output_dir}/{i}.png", crop)
                    i += 1
                    final_bboxes.append(bbox)
                    print(f"Detected object at location {bbox}")

            # make dir
            # os.makedirs(output_dir, exist_ok=True)
            # print('Final bboxed =', len(final_bboxes))

            # grounded resultsSS
            # image_pil = Image.open(input_image_path)
            # image_with_box = plot_boxes_to_image(image_pil, final_bboxes, ['Detected' for idx in final_bboxes])[0]
            # image_with_box.save(os.path.join(f"./{output_dir}/owlvit_box.png"))

            with open('a.txt', 'w') as f:
                for box in final_bboxes:
                    x0, y0, x1, y1 = box
                    x1 = x0 + x1
                    y1 = y0 + y1
                    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                    f.write(f'{shelf_image_class} {query_image_class} {int(x0)} {int(y0)} {int(x1)} {int(y1)}\n')