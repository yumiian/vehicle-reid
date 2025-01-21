import csv

def txt_to_csv(txt_path, csv_path, dataset_type):
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write the CSV header
        csv_writer.writerow(['path', 'id', 'camera'])
        
        # Process each line from the text file
        for line in lines:
            filename = line.strip()
            parts = filename.split('_')
            path = f"{dataset_type}/{filename}"
            id_part = parts[0]  # '0002' 
            camera_part = parts[1]  # 'c002' 

            # Write the row to the CSV
            csv_writer.writerow([path, id_part, camera_part])
