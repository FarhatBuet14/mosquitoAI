from genericpath import isdir
import pandas as pd
import numpy as np
import os 
import shutil


data_folder = "../../../data/Mosquito_AI/PHOTOS/Larvae"
destination_folder = "../../../data/working/img/larvae_localization"
anno_folder = "../../../data/working/annotation/larvae_localization"

img_names = []
source_paths = []
destination_paths = []
modified_names = []
classes = []
number = 0
specimen_number = 0

def file_detected(img_n, img_p, num, gss, spec):
    
    img_names.append(img_n)
    source_paths.append(img_p)
    name = f'{gss[0]}_{gss[1]}_s{spec}_{num}.jpg'
    modified_names.append(name)
    destination_paths.append(os.path.join(destination_folder, name))
    classes.append(f'{gss[0]}_{gss[1]}')
    if(".jpeg" in name): name.replace(".jpeg", "jpg")
    shutil.copy(os.path.join(img_p, img_n), os.path.join(destination_folder, name))
    
    return 0

for f in os.listdir(data_folder): # Source
    if("USF" in f):
        continue
        # for ff in os.listdir(os.path.join(data_folder, f)): # Genus Species
        #     if('.DS_Store' != ff):
        #         if("3rd" not in ff): # Skip 3rd instars
        #             gs = ff.split("_")[0].lower()
        #             g_s =[gs.split(" ")[0], gs.split(" ")[1]]
        #             for fff in os.listdir(os.path.join(data_folder, f, ff)): # Session
        #                 if('.DS_Store' != fff):
        #                     for ffff in os.listdir(os.path.join(data_folder, f, ff, fff)): # Specimens Details
        #                         if('.DS_Store' != ffff):
        #                             for ffffg in os.listdir(os.path.join(data_folder, f, ff, fff, ffff)): # Specimens Folders
        #                                 if('.DS_Store' != ffffg):
        #                                     specimen_number += 1
        #                                     for ffffgg in os.listdir(os.path.join(data_folder, f, ff, fff, ffff, ffffg)): # Images
        #                                         if(('.DS_Store' != ffffgg) and ((".jpg" in ffffgg.lower()) or (".jpeg" in ffffgg.lower()))):
        #                                             number += 1
        #                                             bla = file_detected(ffffgg, os.path.join(data_folder, f, ff, fff, ffff, ffffg), number, g_s, specimen_number)
    
    if("CDC" in f):
        for ff in os.listdir(os.path.join(data_folder, f)): # Genus Species
            if('.DS_Store' != ff):
                print(f"Info --- {ff}")
                gs = ff.split("_")[0].lower()
                g_s =[gs.split(" ")[0], gs.split(" ")[1]]
                for ffff in os.listdir(os.path.join(data_folder, f, ff)): # Specimens Details
                    if('.DS_Store' != ffff):
                        print(f"Session --- {ffff}")
                        for ffffg in os.listdir(os.path.join(data_folder, f, ff, ffff)): # Specimens Folders
                            if('.DS_Store' != ffffg and "HEIC" not in ffffg):
                                print(f"Specimen --- {ffffg}")
                                specimen_number += 1
                                for ffffgg in os.listdir(os.path.join(data_folder, f, ff, ffff, ffffg)): # Images
                                    if(('.DS_Store' != ffffgg) and ((".jpg" in ffffgg.lower()) or (".jpeg" in ffffgg.lower()))):
                                        number += 1
                                        bla = file_detected(ffffgg, os.path.join(data_folder, f, ff, ffff, ffffg), number, g_s, specimen_number)
                            else:
                                print(f"{ffffg} - can't take")
    

    # if("Ethiopia" in f):
    #     for ff in os.listdir(os.path.join(data_folder, f)): # Genus Species Every Info
    #         if('.DS_Store' != ff):
    #             gs = ff.split("__")[1].lower()
    #             g_s =[gs.split(" ")[0], gs.split(" ")[1]]
    #             for fff in os.listdir(os.path.join(data_folder, f, ff)): # Specimens Folders
    #                 if('.DS_Store' != fff):
    #                     specimen_number += 1
    #                     for fffg in os.listdir(os.path.join(data_folder, f, ff, fff)): # Images
    #                         if(('.DS_Store' != fffg) and ((".jpg" in fffg.lower()) or (".jpeg" in fffg.lower()))):
    #                             number += 1
    #                             bla = file_detected(fffg, os.path.join(data_folder, f, ff, fff), number, g_s, specimen_number)

df = pd.DataFrame(list(zip(modified_names, classes, destination_paths, source_paths, img_names)), \
    columns = ["name", "class", "path", "source_path", "source_name"])

df.to_csv(f'{anno_folder}/localization_data.csv')

print("Finished..")
