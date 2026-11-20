import torch
import os
import sys
import argparse
import json
import time
import subprocess
import shlex
from Other.utility_functions import create_path, get_data_size
from Models.options import Options, save_options
from Models.ensemble_utils import (parse_int_tuple, compute_core_extents_dhw,
    expand_extents_dhw, extents_to_string, build_manifest)

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")
# shlex in POSIX mode treats backslashes as escapes, so use forward slashes.
PY_CMD = '"' + sys.executable.replace("\\", "/") + '"'

'''
This file is used to manage multiple jobs that are to be submitted
to the same machine (on OSC, ThetaGPU, etc) that have multiple GPUs.
This code will automatically queue one job per GPU and add another job
to the GPU after the GPU is freed from the previous job.

Settings files to run are in /Code/BatchRunSettings/
'''

# Parses the settings JSON into a list of 
# shell commands to fire off
def build_commands(settings_path, skip_existing=False):
    f = open(settings_path)
    data = json.load(f)
    f.close()
    commands = []
    command_names = []
    log_locations = []
    run_number = 0
    for i in range(len(data)):
        
        
        script_name = data[i][0]
        variables = data[i][1]
        
        if("test" in script_name and "all" == variables['load_from']):
            all_saves = os.listdir(save_folder)
            for fold in all_saves:
                run_name = str(run_number)
                command_names.append(run_name)        
                
                command_string = PY_CMD + " Code/" + str(script_name) + " --load_from " + fold + " " + \
                    "--tests_to_run " + variables['tests'] + " "
                commands.append(command_string)
                
                log_locations.append(os.path.join(save_folder, fold, "test_log.txt"))
                create_path(os.path.join(save_folder, fold))
                
                run_number += 1
        
        # Handle ensemble training specifically
        elif("train" in script_name and "ensemble" in variables.keys() and \
            variables['ensemble']):
            print(f"Ensemble model being trained - creating jobs")
            ensemble_grid = variables['ensemble_grid']
            ensemble_grid = [eval(i) for i in ensemble_grid.split(",")]
            
            full_shape = get_data_size(os.path.join(data_folder, variables['data']))
            print(f"Ensemble grid of {ensemble_grid} for data of size {full_shape}")

            base_opt = Options.get_default()
            for var_name in variables.keys():
                base_opt[var_name] = variables[var_name]
            base_opt['full_shape'] = list(full_shape)
            create_path(os.path.join(save_folder, base_opt['save_name']))
            save_options(base_opt, os.path.join(save_folder, base_opt['save_name']))

            # Geometry lives in Models/ensemble_utils.py. Indices are DHW (storage order);
            # the original variable names x/y/z here referred to D/H/W.
            ghost_cells = base_opt['ensemble_ghost_cells']+1
            manifest = build_manifest(full_shape, ensemble_grid, base_opt['ensemble_ghost_cells'])
            with open(os.path.join(save_folder, base_opt['save_name'], "ensemble_manifest.json"), "w") as mf:
                json.dump(manifest, mf, indent=2)
            print(f"Configured extra ghost cells {base_opt['ensemble_ghost_cells']} -> actual overlap {ghost_cells} cells")

            for x_ind in range(ensemble_grid[0]):
                for y_ind in range(ensemble_grid[1]):
                    for z_ind in range(ensemble_grid[2]):
                        core = compute_core_extents_dhw(full_shape, ensemble_grid, (x_ind, y_ind, z_ind))
                        extents = extents_to_string(expand_extents_dhw(core, full_shape, ghost_cells))
                        run_name = str(run_number)

                        command_names.append(run_name)           
                        command = PY_CMD + " Code/" + str(script_name) + " "
                        
                        for var_name in variables.keys():
                            base_opt[var_name] = variables[var_name]
                            if(var_name == 'save_name'):
                                new_save_name = f"{str(variables[var_name])}/{extents}"
                                command = f"{command} --{str(var_name)} {new_save_name} "
                            elif "ensemble" not in var_name:
                                command = f"{command} --{str(var_name)} {str(variables[var_name])} "
                        command = f"{command} --extents {extents} --grid_index {x_ind},{y_ind},{z_ind} "
                        if(skip_existing and "train" in script_name and
                           os.path.exists(os.path.join(save_folder, new_save_name, "model.ckpt.tar"))):
                            print(f"Skipping {new_save_name}: checkpoint already exists")
                            command_names.pop()
                            run_number += 1
                            continue
                        commands.append(command)

                        if("train" in script_name):
                            if(os.path.exists(os.path.join(save_folder, new_save_name, "train_log.txt"))):
                                os.remove(os.path.join(save_folder, new_save_name, "train_log.txt"))
                            log_locations.append(os.path.join(save_folder, new_save_name, "train_log.txt"))
                            create_path(os.path.join(save_folder, new_save_name))
                        elif("test" in script_name):
                            log_locations.append(os.path.join(save_folder, variables['load_from'], "test_log.txt"))
                            create_path(os.path.join(save_folder, variables["load_from"]))
                        else:
                            log_locations.append(os.path.join(save_folder, variables["load_from"], "log.txt"))
                            create_path(os.path.join(save_folder, variables["load_from"]))
                        run_number += 1

        else:
            
            run_name = str(run_number)
            command_names.append(run_name)           
            command = PY_CMD + " Code/" + str(script_name) + " "
            
            for var_name in variables.keys():
                command = command + "--" + str(var_name) + " "
                command = command + str(variables[var_name]) + " "
            commands.append(command)
            if("train" in script_name):
                if(os.path.exists(os.path.join(save_folder, variables["save_name"], "train_log.txt"))):
                    os.remove(os.path.join(save_folder, variables["save_name"], "train_log.txt"))
                log_locations.append(os.path.join(save_folder, variables["save_name"], "train_log.txt"))
                create_path(os.path.join(save_folder, variables["save_name"]))
            elif("test" in script_name):
                log_locations.append(os.path.join(save_folder, variables['load_from'], "test_log.txt"))
                create_path(os.path.join(save_folder, variables["load_from"]))
            else:
                log_locations.append(os.path.join(save_folder, variables["load_from"], "log.txt"))
                create_path(os.path.join(save_folder, variables["load_from"]))
            run_number += 1
    
    
    return command_names, commands, log_locations

# Parses the string of usable devices
def parse_devices(devices_text):
    devices = devices_text.split(',')
    for i in range(len(devices)):
        devices[i] = devices[i].strip()
        if(devices[i].isnumeric()):
            devices[i] = "cuda:"+str(devices[i])
        else:
            devices[i] = str(devices[i])
    return devices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains models given settings on available gpus')
    parser.add_argument('--settings',default=None,type=str,
        help='The settings file with options for each model to train')
    parser.add_argument('--devices',default="all",type=str,
        help='Which [cuda] devices(s) to train on, separated with commas. Default: all, which uses all available CUDA devices')
    # devices can also be the indices of the GPUS, ex: "0,1,2,4,5,7" or it can simply be "cpu"
    parser.add_argument('--data_devices',default="same",type=str,
        help='Which devices to put the training data on. "same" as model, or "cpu".')
    parser.add_argument('--skip_existing',default="false",type=str,
        help='Skip ensemble bricks that already have model.ckpt.tar (resume a crashed queue).')
    
    args = vars(parser.parse_args())

    settings_path = os.path.join(project_folder_path, "Code", "BatchRunSettings", args['settings'])
    command_names, commands, log_locations = build_commands(settings_path,
        skip_existing=args['skip_existing'].lower() in ('1', 'true', 'yes'))

    if(args['devices'] == "all"):
        available_devices = []
        for i in range(torch.cuda.device_count()):
            available_devices.append("cuda:" + str(i))
        if(len(available_devices)==0):
            available_devices.append("cpu")
            
    else:
        available_devices = parse_devices(args['devices'])

    total_jobs = len(commands)
    jobs_training = []
    while(len(commands) + len(jobs_training) > 0):
        # Check if any jobs have finished and a GPU is freed
        i = 0 
        while i < len(jobs_training):
            c_name, job, gpu, job_start_time = jobs_training[i]
            job_code = job.poll()
            if(job_code is not None):
                # Job has finished executing
                jobs_training.pop(i)
                job_end_time = time.time()
                print(f"Job {c_name} on {gpu} has finished with exit code {job_code} after {(job_end_time-job_start_time)/60 : 0.02f} minutes")
                # The gpu is freed, added back to available_devices
                available_devices.append(gpu)
            else:
                i += 1

        # Check if any gpus are available for commands in queue
        if(len(available_devices) > 0 and len(commands)>0):
            c = commands.pop(0)
            c_name = command_names.pop(0)
            log_location = log_locations.pop(0)
            g = str(available_devices.pop(0))
            if(args['data_devices'] == "same"):
                data_device = str(g)
            else:
                data_device = "cpu"
            c = c + "--device " + g + " --data_device " + data_device
            c_split = shlex.split(c)
            # Logging location
            output_path = open(log_location,'a+')
            # Start the job
            print(f"Starting job {c_name}/{total_jobs} on device {g}")
            job = subprocess.Popen(c_split, stdout=output_path, stderr=output_path)
            jobs_training.append((c_name, job, g, time.time()))
        else:
            # Otherwise wait
            time.sleep(1.0)

    print("All jobs have completed.")
    quit()
