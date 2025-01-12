import csv
# import xml.etree.ElementTree as ET

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

# def xml_to_csv(xml_path, csv_path, dataset_type):
#     with open(xml_path, 'r', encoding='gb2312') as file:
#         # Read the content of the file
#         xml_data = file.read()

#     # Parse the XML data from the string
#     root = ET.fromstring(xml_data)

#     with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file)
#         # Write header row
#         writer.writerow(['path', 'id', 'camera'])
        
#         # Iterate over each <Item> element
#         for item in root.find(".//Items").iter("Item"):
#             # Extract attributes
#             image_name = item.attrib['imageName']
#             vehicle_id = item.attrib['vehicleID']
#             camera_id = item.attrib['cameraID']
            
#             # Write data to the CSV
#             path = f"{dataset_type}/{image_name}"
#             writer.writerow([path, vehicle_id, camera_id])