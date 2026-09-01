from .settings import *
import os
import time

'''
params setting
'''
settings = JsonConfig()
settings.load_json_file("train_settings.json")

decoder_only_training = settings.decoder_only_training

project_name = settings.project_name
dataset_path = settings.dataset_path
epoch_number = settings.epoch_number
batch_size = settings.batch_size
train_continue = settings.train_continue
train_continue_start = settings.train_continue_start
discriminator_weight = settings.discriminator_weight
encoder_weight = settings.encoder_mse_weight
encoder_ssim_weight = settings.encoder_ssim_weight
encoder_lpips_weight = settings.encoder_lpips_weight
decoder_weight = settings.decoder_weight
train_continue_path = settings.train_continue_path
train_continue_epoch = settings.train_continue_epoch
train_continue_from_lastest_epoch = settings.train_continue_from_lastest_epoch
save_images_number = settings.save_images_number
lr = settings.lr
H, W, message_length = settings.H, settings.W, settings.message_length,
noise_layers = settings.noise_layers
encoder_start_fn = settings.encoder_start
encoder_peak_fn = settings.encoder_peak
seed = settings.seed


'''
file preparing
'''
full_project_name = project_name + "_m" + str(message_length)
for noise in noise_layers:
	full_project_name += "_" + "_"

def initialize_training_folders():
	"""Only call this at the start of train.py to create result folders"""
	global result_folder
	result_folder = "results/" + time.strftime(full_project_name + "__%Y_%m_%d__%H_%M_%S", time.localtime()) + "/"
	if not os.path.exists(result_folder): os.mkdir(result_folder)
	if not os.path.exists(result_folder + "images/"): os.mkdir(result_folder + "images/")
	if not os.path.exists(result_folder + "models/"): os.mkdir(result_folder + "models/")
	with open(result_folder + "/train_params.txt", "w") as file:
		content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",
															time.localtime()) + "-----------------------\n"

		for item in settings.get_items():
			content += item[0] + " = " + str(item[1]) + "\n"

		print(content)
		file.write(content)

	with open(result_folder + "/train_log.txt", "w") as file:
		content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",
															time.localtime()) + "-----------------------\n"
		file.write(content)
	with open(result_folder + "/val_log.txt", "w") as file:
		content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",
															time.localtime()) + "-----------------------\n"
		file.write(content)
	return result_folder

# Initialize with a placeholder until initialize_training_folders() is called
result_folder = "results/placeholder/"
