import albumentations as A
import cv2
import os

def get_abbreviation(transform_list):
    """Generate abbreviation for output filename based on selected augmentations."""
    abbr = ""
    for transform in transform_list:
        if isinstance(transform, A.HorizontalFlip):
            abbr += "HF_"
        elif isinstance(transform, A.VerticalFlip):
            abbr += "VF_"
        elif isinstance(transform, A.RandomRotate90):
            abbr += "RR90_"
        elif isinstance(transform, A.Erasing):
            abbr += "ERS_"
        elif isinstance(transform, A.ColorJitter):
            abbr += "CJ_"
        elif isinstance(transform, A.Sharpen):
            abbr += "SHRP_"
        elif isinstance(transform, A.UnsharpMask):
            abbr += "UNSM_"
        elif isinstance(transform, A.CLAHE):
            abbr += "CLAHE_"
    return abbr

def augment(transform_list, seed, image_path):
    transform = A.Compose(transform_list, seed=seed, strict=True)
    abbr = get_abbreviation(transform_list)

    for filename in os.listdir(image_path):
        if not filename.endswith((".png", ".jpg", ".jpeg")):
            continue

        filepath = os.path.join(image_path, filename)

        image = cv2.imread(filepath, cv2.COLOR_BGR2RGB)
        augmented_image = transform(image=image)["image"]
        
        output_filename = os.path.join(image_path, f"{abbr}{filename}")
        cv2.imwrite(output_filename, augmented_image)