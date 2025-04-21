import os
import json

import xmltodict

def parse_xml_to_dict(xml_path):
    """Parse an XML file and convert it to a dictionary."""

    try:
       the_dict = xmltodict.parse(open(xml_path,encoding="utf-8").read()) 
       return the_dict
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return None

def process_data_folder(data_folder):
    """Loop through each subfolder in the data folder and process gamelist.xml."""

    for subfolder in os.listdir(data_folder):
        subfolder_path = os.path.join(data_folder, subfolder)
        if os.path.isdir(subfolder_path):
            gamelist_path = os.path.join(subfolder_path, "gamelist.xml")
            if os.path.exists(gamelist_path) and subfolder == 'arcade':
                print(f"Processing {gamelist_path}...")
                gamelist_dict = parse_xml_to_dict(gamelist_path)
                if gamelist_dict:
                    output_file = os.path.join(subfolder_path, "gamelist.json")
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(gamelist_dict, f, indent=4, ensure_ascii=False)
                    print(f"Serialized dictionary to {output_file}")
            else:
                print(f"No gamelist.xml found in {subfolder_path}")

if __name__ == "__main__":

    from dotenv import load_dotenv, find_dotenv
    load_dotenv() # read local .env file

    import warnings
    warnings.filterwarnings('ignore')

    data_folder = "data"  # Replace with the path to your data folder

    process_data_folder(data_folder)
