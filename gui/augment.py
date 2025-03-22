import albumentations as A
import cv2
import os

def augment(transform_list, seed, image_path):
    transform = A.Compose(transform_list, seed=seed, strict=True)

    for filename in os.listdir(image_path):
        if not filename.endswith((".png", ".jpg", ".jpeg")):
            continue

        filepath = os.path.join(image_path, filename)

        image = cv2.imread(filepath, cv2.COLOR_BGR2RGB)
        augmented_image = transform(image=image)["image"]
        
        output_filename = os.path.join(image_path, f"_{filename}")
        cv2.imwrite(output_filename, augmented_image)