import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from Image_Segmentation.evaluation import *
from Image_Segmentation.network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import csv
import re
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import transforms as T

class Solver(object):
	def __init__(self, config, train_loader= None, valid_loader= None, test_loader= None, model_path_eval= None):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader
		self.model_path_eval = model_path_eval

		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = torch.nn.BCELoss()
		self.augmentation_prob = config.augmentation_prob

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet = U_Net(img_ch=3,output_ch=1)
		elif self.model_type =='R2U_Net':
			self.unet = R2U_Net(img_ch=3,output_ch=1,t=self.t)
		elif self.model_type =='AttU_Net':
			self.unet = AttU_Net(img_ch=3,output_ch=1)
		elif self.model_type == 'R2AttU_Net':
			self.unet = R2AttU_Net(img_ch=3,output_ch=1,t=self.t)
			

		self.optimizer = optim.Adam(list(self.unet.parameters()),
									  self.lr, [self.beta1, self.beta2])
		self.unet.to(self.device)

		# self.print_network(self.unet, self.model_type)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()

	def compute_accuracy(self,SR,GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)

		acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img


	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#
		
		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))

		# U-Net Train
		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
		else:
			# Train for Encoder
			lr = self.lr
			best_unet_score = 0.
			
			for epoch in range(self.num_epochs):

				self.unet.train(True)
				epoch_loss = 0
				
				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length = 0

				for i, (images, GT, _) in enumerate(self.train_loader):
					# GT : Ground Truth

					images = images.to(self.device)
					GT = GT.to(self.device)

					# SR : Segmentation Result
					SR = self.unet(images)
					SR_probs = F.sigmoid(SR)
					SR_flat = SR_probs.view(SR_probs.size(0),-1)

					GT_flat = GT.view(GT.size(0),-1)
					loss = self.criterion(SR_flat,GT_flat)
					epoch_loss += loss.item()

					# Backprop + optimize
					self.reset_grad()
					loss.backward()
					self.optimizer.step()

					acc += get_accuracy(SR,GT)
					SE += get_sensitivity(SR,GT)
					SP += get_specificity(SR,GT)
					PC += get_precision(SR,GT)
					F1 += get_F1(SR,GT)
					JS += get_JS(SR,GT)
					DC += get_DC(SR,GT)
					length += images.size(0)

				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length

				# Print the log info
				print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
					  epoch+1, self.num_epochs, \
					  epoch_loss,\
					  acc,SE,SP,PC,F1,JS,DC))

			

				# Decay learning rate
				if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
					lr -= (self.lr / float(self.num_epochs_decay))
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = lr
					print ('Decay learning rate to lr: {}.'.format(lr))
				
				
				#===================================== Validation ====================================#
				self.unet.train(False)
				self.unet.eval()

				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length=0
				for i, (images, GT, _) in enumerate(self.valid_loader):

					images = images.to(self.device)
					GT = GT.to(self.device)
					SR = F.sigmoid(self.unet(images))
					acc += get_accuracy(SR,GT)
					SE += get_sensitivity(SR,GT)
					SP += get_specificity(SR,GT)
					PC += get_precision(SR,GT)
					F1 += get_F1(SR,GT)
					JS += get_JS(SR,GT)
					DC += get_DC(SR,GT)
						
					length += images.size(0)
					
				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length
				unet_score = JS + DC

				print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f'%(acc,SE,SP,PC,F1,JS,DC))
				
				'''
				torchvision.utils.save_image(images.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_image.png'%(self.model_type,epoch+1)))
				torchvision.utils.save_image(SR.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
				torchvision.utils.save_image(GT.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
				'''


				# Save Best U-Net model
				if unet_score > best_unet_score:
					best_unet_score = unet_score
					best_epoch = epoch
					best_unet = self.unet.state_dict()
					print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
					torch.save(best_unet,unet_path)
					
			#===================================== Test ====================================#
			del self.unet
			del best_unet
			self.build_model()
			self.unet.load_state_dict(torch.load(unet_path))
			
			self.unet.train(False)
			self.unet.eval()

			acc = 0.	# Accuracy
			SE = 0.		# Sensitivity (Recall)
			SP = 0.		# Specificity
			PC = 0. 	# Precision
			F1 = 0.		# F1 Score
			JS = 0.		# Jaccard Similarity
			DC = 0.		# Dice Coefficient
			length=0
			for i, (images, GT, _) in enumerate(self.test_loader):

				images = images.to(self.device)
				GT = GT.to(self.device)
				SR = F.sigmoid(self.unet(images))
				acc += get_accuracy(SR,GT)
				SE += get_sensitivity(SR,GT)
				SP += get_specificity(SR,GT)
				PC += get_precision(SR,GT)
				F1 += get_F1(SR,GT)
				JS += get_JS(SR,GT)
				DC += get_DC(SR,GT)
						
				length += images.size(0)
					
			acc = acc/length
			SE = SE/length
			SP = SP/length
			PC = PC/length
			F1 = F1/length
			JS = JS/length
			DC = DC/length
			unet_score = JS + DC


			f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
			wr = csv.writer(f)
			wr.writerow([self.model_type,acc,SE,SP,PC,F1,JS,DC,self.lr,best_epoch,self.num_epochs,self.num_epochs_decay,self.augmentation_prob])
			f.close()

	def eval_test(self):
		if os.path.isfile(self.model_path_eval):
			pattern = r"(\w+)-(\d+)-([\d.]+)-(\d+)-([\d.]+)\.pkl"
			# Search for the pattern in the filename
			match = re.search(pattern, self.model_path_eval)
			if match:
				self.model_type = match.group(1)
				self.num_epochs = int(match.group(2))
				self.lr = float(match.group(3))
				self.num_epochs_decay = int(match.group(4))
				self.augmentation_prob = float(match.group(5))



			print(self.model_path_eval)
			self.build_model()
			self.unet.load_state_dict(torch.load(self.model_path_eval))
			print('%s is Successfully Loaded from %s'%(self.model_type,self.model_path_eval))
			self.unet.train(False)
			self.unet.eval()

			acc = 0.	# Accuracy
			SE = 0.		# Sensitivity (Recall)
			SP = 0.		# Specificity
			PC = 0. 	# Precision
			F1 = 0.		# F1 Score
			JS = 0.		# Jaccard Similarity
			DC = 0.		# Dice Coefficient
			length=0
			metrics = {
				'filename': [],
				'accuracy': [],
				'sensitivity': [],
				'specificity': [],
				'precision': [],
				'F1_score': [],
				'Jaccard_similarity': [],
				'Dice_coefficient': []
				}


			for i, (images, GT, image_path) in enumerate(self.train_loader):

				images = images.to(self.device)
				GT = GT.to(self.device)
				SR = F.sigmoid(self.unet(images))
				acc += get_accuracy(SR,GT)
				SE += get_sensitivity(SR,GT)
				SP += get_specificity(SR,GT)
				PC += get_precision(SR,GT)
				F1 += get_F1(SR,GT)
				JS += get_JS(SR,GT)
				DC += get_DC(SR,GT)
						
				length += images.size(0)
				

				#save data Collect metrics for each image
				
				metrics['filename'].append(image_path)
				metrics['accuracy'].append(acc/length)
				metrics['sensitivity'].append(SE/length)
				metrics['specificity'].append(SP/length)
				metrics['precision'].append(PC/length)
				metrics['F1_score'].append(F1/length)
				metrics['Jaccard_similarity'].append(JS/length)
				metrics['Dice_coefficient'].append(DC/length)
				print(image_path,acc/length,DC/length)

				with open(os.path.join(self.result_path,'evaluation_metrics.csv'), mode='w', newline='') as file:
					writer = csv.writer(file)
					writer.writerow(['filename', 'accuracy', 'sensitivity', 'specificity', 'precision', 'F1_score', 'Jaccard_similarity', 'Dice_coefficient'])
					for i in range(len(metrics['filename'])):
						writer.writerow([metrics['filename'][i], metrics['accuracy'][i], metrics['sensitivity'][i], metrics['specificity'][i],
										metrics['precision'][i], metrics['F1_score'][i], metrics['Jaccard_similarity'][i], metrics['Dice_coefficient'][i]])

				images_cpu = images.detach().cpu().numpy()
				SR_cpu = SR.detach().cpu().numpy()
				GT_cpu = GT.detach().cpu().numpy()
				
				# Plot the images, model output, and ground trut
				for idx in range(images_cpu.shape[0]):
					fig, axs = plt.subplots(1, 3, figsize=(15, 5))

					# Assuming the images are single channel (grayscale), if not adjust accordingly
					axs[0].imshow(images_cpu[idx].transpose(1, 2, 0))
					axs[0].set_title('Input Image')
					
					axs[1].imshow(SR_cpu[idx][0], cmap='gray')
					axs[1].set_title('Model Output')

					axs[2].imshow(GT_cpu[idx][0], cmap='gray')
					axs[2].set_title('Ground Truth')

					plt.show()
				

			acc = acc/length
			SE = SE/length
			SP = SP/length
			PC = PC/length
			F1 = F1/length
			JS = JS/length
			DC = DC/length
			unet_score = JS + DC
			print([self.model_type,acc,SE,SP,PC,F1,JS,DC,self.lr,self.num_epochs,self.num_epochs_decay,self.augmentation_prob])
		
		else:
			print("path does not exit")


	def eval_image(self, image_path):
		if os.path.isfile(self.model_path_eval):
			pattern = r"(\w+)-(\d+)-([\d.]+)-(\d+)-([\d.]+)\.pkl"
			match = re.search(pattern, self.model_path_eval)
			if match:
				self.model_type = match.group(1)
				self.num_epochs = int(match.group(2))
				self.lr = float(match.group(3))
				self.num_epochs_decay = int(match.group(4))
				self.augmentation_prob = float(match.group(5))

			print(self.model_path_eval)
			self.build_model()
			self.unet.load_state_dict(torch.load(self.model_path_eval))
			print('%s is Successfully Loaded from %s' % (self.model_type, self.model_path_eval))
			self.unet.train(False)
			self.unet.eval()

			# Load the specific image
			
			image_org = Image.open(image_path).convert("RGB")
		
			Transform = []
			aspect_ratio = image_org.size[1] / image_org.size[0]
			Transform.append(T.Resize((int(256*aspect_ratio)-int(256*aspect_ratio)%16,256)))
			Transform.append(T.ToTensor())
			Transform = T.Compose(Transform)
			
			image = Transform(image_org)

			Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			image = Norm_(image)
			device = self.device if hasattr(self, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
			image = image.to(device)
			image = image.unsqueeze(0)
			seg = self.unet(image)
			print(seg.shape)


			
			seg = seg.squeeze(0)

			# Convert the tensor to a numpy array and transpose the dimensions to (height, width, channels)
			seg_array = seg.permute(1, 2, 0).cpu().detach().numpy()



			fig, ax = plt.subplots(1, 2, figsize=(12, 6))

			# Original Image
			ax[0].imshow(image_org)
			ax[0].set_title('Original Image')
			ax[0].axis('off')

			# Segmentation Result
			binary_mask = (seg_array > 0).astype(np.uint8)
			ax[1].imshow(binary_mask, cmap='gray')
			ax[1].set_title('Segmentation Result')
			ax[1].axis('off')

			plt.show()

