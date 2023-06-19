import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from one_shot_classification import *

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def crop_image_with_xywh(image, bbox_xywh):
    # Extract the bounding box coordinates
    x, y, w, h = bbox_xywh

    # Calculate the coordinates for cropping
    left = x
    top = y
    right = x + w
    bottom = y + h

    # Crop the image
    cropped_image = image[top:bottom, left:right]

    return cropped_image

def crop_image_with_mask(image, mask, bbox):
    image_new = image * mask[:, :, None]

    # Crop the image based on the bounding box
    cropped_image = crop_image_with_xywh(image_new, bbox)

    return cropped_image

def get_crops_and_bboxes(image, masks):
  crops_and_bboxes = []
  bboxes_segmentations =  []
  for mask in masks:
    bboxes_segmentations.append((mask[ 'bbox'], mask['segmentation']))

  for bbox, segment in bboxes_segmentations:
    cropped_image = crop_image_with_mask(image, segment, bbox)
    crops_and_bboxes.append((cropped_image, bbox))

  return crops_and_bboxes

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def plot_boxes_to_image(image_pil, boxes, labels):
    # H, W = image.size[1], image.size[0]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x1 = x0 + x1
        y1 = y0 + y1
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Segment Aything with siamese classification", add_help=True)

    parser.add_argument("--image_path", "-i", type=str, required=True, help="path to image file")
    parser.add_argument("--query_image_path", "-qi", type=str, default="", required=True, help="path to query image file")
    
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )
    parser.add_argument("--saved_model_dir", type=str, default='',
                        help="Where restore model parameters from.")
    args = parser.parse_args()

    input_image_path = args.image_path
    query_image_path = args.query_image_path
    saved_model_dir = args.saved_model_dir
    output_dir = args.output_dir
    
    image = cv2.imread(input_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    query_image = cv2.imread(query_image_path)
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

    sam_checkpoint = "./sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam# .to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    masks = mask_generator.generate(image)

    crops_and_bboxes = get_crops_and_bboxes(image, masks)

     ## reset tf graph
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

    query_embed = get_embeddings(query_image, model_siamese)

    final_bboxes = []
    i = 0
    print('Totoal segments =', len(crops_and_bboxes))
    for crop, bbox in crops_and_bboxes:
        if 0 in crop.shape:
            continue
        if one_shot_match(crop, query_embed, model_siamese, from_query_embed = True):
            # cv2.imwrite(f"./{output_dir}/{i}.png", crop)
            i += 1
            final_bboxes.append(bbox)
            print(f"Detected object at location {bbox}")

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    print('Final bboxed =', len(final_bboxes))

    # grounded resultsSS
    image_pil = Image.open(input_image_path)
    image_with_box = plot_boxes_to_image(image_pil, final_bboxes, ['Detected' for idx in final_bboxes])[0]
    image_with_box.save(os.path.join(f"./{output_dir}/owlvit_box.png"))