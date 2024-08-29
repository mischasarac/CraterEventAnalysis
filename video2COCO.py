#   Code here is taken from https://github.com/NWalker4483/LabelStudioVideo/blob/main/label_studio2coco.py
#   Originally produced by NWalker4483

import argparse
import json
import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import shutil

def parse_video_name(annotation):
    return Path(annotation['url']).stem.split('-')[1]

def match_video_file(annotation, video_dir):
    url = annotation['url']
    video_basename = os.path.basename(url)
    video_name = video_basename.split('-')[1]  # Assuming the format is always 'something-videoname.mp4'
    
    for file in os.listdir(video_dir):
        if file.startswith(video_name):
            return os.path.join(video_dir, file)
    
    print(f"Warning: No matching video file found for {video_name}")
    return None

def linear_interpolation(start_seq, end_seq, label, start_frame, end_frame):
    frame_diff = end_frame - start_frame
    
    interpolated_boxes = {}
    
    for frame in range(start_frame, end_frame + 1):
        if frame == start_frame:
            t = 0
        elif frame == end_frame:
            t = 1
        else:
            t = (frame - start_frame) / frame_diff
        
        interpolated_box = {
            'label': label,
            'x': start_seq['x'] + t * (end_seq['x'] - start_seq['x']),
            'y': start_seq['y'] + t * (end_seq['y'] - start_seq['y']),
            'width': start_seq['width'] + t * (end_seq['width'] - start_seq['width']),
            'height': start_seq['height'] + t * (end_seq['height'] - start_seq['height'])
        }
        
        interpolated_boxes[frame] = interpolated_box
    
    return interpolated_boxes

def process_video_annotation(video_annotation, video_dir, output_base, frame_sample_rate, labels_dict, image_id, annotation_id, label_studio_fps):
    video_path = match_video_file(video_annotation, video_dir) if video_dir else None
    video_name = parse_video_name(video_annotation)
    
    print(f"Processing annotation for video: {video_name}")
    
    if video_path:
        vidcap = cv2.VideoCapture(str(video_path))
        actual_fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_rate_ratio = actual_fps / label_studio_fps if label_studio_fps else 1
    else:
        frame_rate_ratio = 1
    
    total_frames = max(int(seq['frame'] * frame_rate_ratio) for subject in video_annotation['box'] for seq in subject['sequence'])
    
    # Gather labels from this video
    for subject in video_annotation['box']:
        for label in subject['labels']:
            if label not in labels_dict:
                labels_dict[label] = len(labels_dict)
    
    boxes_dict = {frame: [] for frame in range(1, total_frames + 1)}
    
    for subject in video_annotation['box']:
        subject_labels = subject['labels']
        
        if len(subject_labels) == 1:
            label = labels_dict[subject_labels[0]]
        else:
            raise ValueError("Each subject must have exactly one label.")
        
        sequences = sorted(subject['sequence'], key=lambda x: x['frame'])
        
        for i, seq in enumerate(sequences):
            frame = int(seq['frame'] * frame_rate_ratio)
            
            if seq['enabled']:
                next_seq = sequences[i+1] if i+1 < len(sequences) else None
                
                if next_seq:
                    end_frame = int(next_seq['frame'] * frame_rate_ratio) - 1
                else:
                    end_frame = total_frames
                
                current_box = {k: float(seq[k]) for k in ('x', 'y', 'width', 'height')}
                if next_seq and next_seq['enabled']:
                    next_box = {k: float(next_seq[k]) for k in ('x', 'y', 'width', 'height')}
                else:
                    next_box = current_box
                
                interpolated_boxes = linear_interpolation(current_box, next_box, label, frame, end_frame)
                for f, box in interpolated_boxes.items():
                    boxes_dict[f].append(box)
            else:
                box = {'label': label, 'x': float(seq['x']), 'y': float(seq['y']), 'width': float(seq['width']), 'height': float(seq['height'])}
                boxes_dict[frame].append(box)
    
    print(f"Exporting annotations in COCO format for {video_name}")
    
    output_path = Path(output_base)
    temp_images_dir = output_path / 'temp_images'
    temp_images_dir.mkdir(parents=True, exist_ok=True)
    
    coco_images = []
    coco_annotations = []
    
    padding = 8  # Adjust this value if you expect more than 99,999,999 frames
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) if video_path else total_frames

    with tqdm(total=total_frames, desc=f"Processing {video_name}") as pbar:
        for current_frame in range(1, total_frames + 1):
            if video_path and current_frame % frame_sample_rate == 0:
                success, image = vidcap.read()
                if success:
                    image_filename = f'{image_id:0{padding}d}.jpg'
                    cv2.imwrite(str(temp_images_dir / image_filename), image)
                    
                    height, width = image.shape[:2]
                    coco_images.append({
                        "id": image_id,
                        "file_name": image_filename,
                        "height": height,
                        "width": width
                    })
                    
                    for box in boxes_dict[current_frame]:
                        x = box['x'] * width / 100
                        y = box['y'] * height / 100
                        w = box['width'] * width / 100
                        h = box['height'] * height / 100
                        
                        coco_annotations.append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": box['label'],
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
                        annotation_id += 1
                    
                    image_id += 1
                else:
                    print(f"Warning: Could not read frame {current_frame} from video.")
            elif video_path:
                # Skip frames that we're not saving
                vidcap.grab()
            
            pbar.update(1)
    
    if video_path:
        vidcap.release()

    print(f"Finished processing {video_name}")
    return labels_dict, image_id, annotation_id, coco_images, coco_annotations

def split_data(coco_images, coco_annotations, output_base, split_ratios):
    all_image_ids = [img['id'] for img in coco_images]
    random.shuffle(all_image_ids)

    total_images = len(all_image_ids)
    split_points = [int(ratio * total_images) for ratio in np.cumsum(split_ratios)[:-1]]

    split_ids = np.split(all_image_ids, split_points)
    splits = dict(zip(['train', 'val', 'test'][:len(split_ratios)], [set(ids) for ids in split_ids]))

    split_data = {split: {"images": [], "annotations": []} for split in splits}

    output_path = Path(output_base)
    temp_images_dir = output_path / 'temp_images'
    images_dir = output_path / 'images'

    for split in splits:
        (images_dir / f'{split}').mkdir(parents=True, exist_ok=True)

    for img in coco_images:
        for split, ids in splits.items():
            if img['id'] in ids:
                split_data[split]["images"].append(img)
                # Move the image to the corresponding split folder
                src = temp_images_dir / img['file_name']
                dst = images_dir / f'{split}' / img['file_name']
                shutil.move(src, dst)
                break

    for ann in coco_annotations:
        for split, ids in splits.items():
            if ann['image_id'] in ids:
                split_data[split]["annotations"].append(ann)
                break

    # Remove the temporary images directory
    shutil.rmtree(temp_images_dir)

    print("Data split complete. " + ", ".join(f"{split.capitalize()}: {len(ids)}" for split, ids in splits.items()))
    return split_data

def main(json_path, video_dir, output_base, frame_sample_rate, split_ratios, label_studio_fps):
    print("Parsing annotations from JSON")
    with open(json_path) as f:
        video_annotations = json.load(f)

    labels_dict = {}
    image_id = 0
    annotation_id = 0
    all_coco_images = []
    all_coco_annotations = []

    for video_annotation in video_annotations:
        labels_dict, image_id, annotation_id, coco_images, coco_annotations = process_video_annotation(
            video_annotation, video_dir, output_base, frame_sample_rate, labels_dict, image_id, annotation_id, label_studio_fps
        )
        all_coco_images.extend(coco_images)
        all_coco_annotations.extend(coco_annotations)
    
    # Create COCO categories
    coco_categories = [{"id": id, "name": name} for name, id in labels_dict.items()]
    
    # Split data
    split_data_dict = split_data(all_coco_images, all_coco_annotations, output_base, split_ratios)
    
    # Save COCO JSON files
    output_path = Path(output_base)
    annotations_dir = output_path / 'annotations'
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    for split, data in split_data_dict.items():
        coco_data = {
            "images": data["images"],
            "annotations": data["annotations"],
            "categories": coco_categories
        }
        with open(annotations_dir / f'instances_{split}.json', 'w') as f:
            json.dump(coco_data, f)
    
    print("All videos processed successfully.")
    print(f"Final list of classes: {', '.join(sorted(labels_dict.keys()))}")

    print("\nDataset structure:")
    print(f"{output_base}/")
    print("├── annotations/")
    for split in split_data_dict.keys():
        print(f"│   ├── instances_{split}.json")
    print("└── images/")
    for split in split_data_dict.keys():
        print(f"    ├── {split}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""This script processes video annotations exported from Label Studio
        in JSON-MIN format, converting them directly into the COCO dataset format. 
        The script supports interpolation of bounding boxes for intermediate frames based 
        on key-frame annotations and exports these labels along with corresponding frames.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-j", "--json_path", required=True, help="Path to JSON annotations")
    parser.add_argument("-v", "--video_dir", help="Path to directory containing video files")
    parser.add_argument("-o", "--output_base", default='coco_dataset/', help="Path to output base directory")
    parser.add_argument("-f", "--frame_sample_rate", type=int, default=60, help="Save every Nth frame (e.g., 60 to save every 60th frame)")
    parser.add_argument("-r", "--split_ratios", default="0.7,0.15,0.15", help="Comma-separated list of ratios for train,val,test splits")
    parser.add_argument("--label_studio_fps", type=float, help="Frame rate used in Label Studio (default: None, meaning use video's actual frame rate)")
    args = parser.parse_args()

    # Parse and validate split ratios
    split_ratios = [float(r) for r in args.split_ratios.split(',')]
    if not np.isclose(sum(split_ratios), 1.0):
        parser.error(f"The sum of split ratios must be 1.0. Current sum: {sum(split_ratios)}")
    if len(split_ratios) < 1 or len(split_ratios) > 3:
        parser.error("You must provide 1, 2, or 3 split ratios")

    main(args.json_path, args.video_dir, args.output_base, args.frame_sample_rate, 
         split_ratios, args.label_studio_fps)