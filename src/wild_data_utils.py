import os
import cv2
import xml.etree.ElementTree as ET

def get_wild_species_names(annotations_dir):
    species_names = set({})
    
    for annotation_index, annotation_file in enumerate(os.listdir(annotations_dir)):
        # Read xml text
        with open(os.path.join(annotations_dir, annotation_file), "r") as f:
            xml_text = f.read()
        
        name = xml_text[xml_text.index("<name>")+6:xml_text.index("</name>")]
        species_names.add(name)

    return list(species_names)

def load_wild_img_data(species_name, max_imgs, images_dir, annotations_dir):
    imgs = []
    img_bboxes = []
    img_patches = []
    p_within_bounds = []
    num_imgs = 0
    for annotation_index, annotation_file in enumerate(os.listdir(annotations_dir)):

        # Read xml text
        with open(os.path.join(annotations_dir, annotation_file), "r") as f:
            xml_text = f.read()
        
        # Process for certain species name
        name = xml_text[xml_text.index("<name>")+6:xml_text.index("</name>")]
        
        # Add if image is of specified species
        if name == species_name:
            if num_imgs >= max_imgs:
                break

            num_imgs += 1

            filename = xml_text[xml_text.index("<filename>")+10:xml_text.index("</filename>")]
            img = cv2.imread(os.path.join(images_dir, filename))
            imgs.append(img)

            # Load xml
            root = ET.parse(os.path.join(annotations_dir, annotation_file)).getroot()

            # Get list of bounding boxes
            bboxes = []
            for elem in root[5:]:
                bboxes.append([int(elem[-1][0].text), int(elem[-1][1].text), int(elem[-1][2].text), int(elem[-1][3].text)])
            img_bboxes.append(bboxes)

    return imgs, img_bboxes