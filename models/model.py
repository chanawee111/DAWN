from . import *
from .discriminator_network import PatchDiscriminator
from .Encoder_Decoder import Encoder_Decoder 
from .noise_layers import *

class Model:
    
    def __init__(self, image_height, image_width, message_length, noise_layers, device, batch_size, lr, decoder_only_training=False, seed: int = None):

        # if seed is not None:
        #     self.set_seed(seed)

        self.device = device
        self.ramp_function = get_warmup_decoder_weights
        self.prev_decoder_loss = None
        self.prev_encoder_weight = None

        # self.strength_factor = 1.0

        # store weights for validation
        # self.encoder_weight = None
        # self.decoder_weight = None
        # self.discriminator_weight = None

        self.encoder_decoder = Encoder_Decoder(image_height, image_width, message_length, noise_layers,batch_size, device=device).to(device)

        self.discriminator = PatchDiscriminator().to(device)

        self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
        self.discriminator = torch.nn.DataParallel(self.discriminator)

        if decoder_only_training:
            # freeze encoder
            for p in self.encoder_decoder.module.encoder.parameters():
                p.requires_grad = True

        self.optimizer_encoder_decoder = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr
        )

        self.optimizer_discriminator = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr
        )



        # define loss functions
        self.MSE_loss = YCbCrWeightedMSELoss(y_weight=1.0, chroma_weight=1.0)
        self.LPIPS_loss = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
        self.SSIM_loss = SSIM(data_range=2.0, size_average=True, channel=3)
        # edge loss compute 
        self.combined_edge_loss = EdgePenaltyLoss(
                                                edge_width=2,
                                                alpha=1.0,
                                                beta=0.5,
                                                loss_type='combined'
                                                )

        # spatial loss weights
        self.ssim_weight = encoder_ssim_weight
        self.lpips_weight = encoder_lpips_weight
        self.edge_weight = 1.0
        self.BCE_loss = nn.BCEWithLogitsLoss().to(device)
        

        # define weights for encoder_decoder and discriminator
        self.discriminator_weight = discriminator_weight
        self.encoder_weight = encoder_weight
        self.decoder_weight = decoder_weight

    def train(self, image: torch.Tensor, message: torch.Tensor, epoch: int):
        self.encoder_decoder.train()
        self.discriminator.train()

        with torch.enable_grad():
            
            images, messages = image.to(self.device), message.to(self.device)
            encoded_images, noised_image, decoded_message = self.encoder_decoder(images, messages, epoch=epoch)
            # encoded_images = image + (encoded_images - image) * self.strength_factor

            # ------------------------------------------------------------------------
            # discriminator training
            # ------------------------------------------------------------------------
            self.optimizer_discriminator.zero_grad()

            #  RAW : expexted label for image should be "cover" : "1"
            disc_label_cover = self.discriminator(images)
            # disc_cover_loss = self.BCE_loss(disc_label_cover, self.label_cover[:disc_label_cover.size(0)])
            label_cover = torch.ones_like(disc_label_cover, device=self.device)
            disc_cover_loss = self.BCE_loss(disc_label_cover, label_cover)
            disc_cover_loss.backward()

            # GAN : expected label for image should be "encoded" : "0"
            disc_label_encoded = self.discriminator(encoded_images.detach())
            # disc_encoded_loss = self.BCE_loss(disc_label_encoded, self.label_encoded[:disc_label_encoded.size(0)])
            label_encoded = torch.zeros_like(disc_label_encoded, device=self.device)
            disc_encoded_loss = self.BCE_loss(disc_label_encoded, label_encoded)
            disc_encoded_loss.backward()

            self.optimizer_discriminator.step()

            # ------------------------------------------------------------------------
            # Encoder training and decoder training
            # ------------------------------------------------------------------------
            self.optimizer_encoder_decoder.zero_grad()

            # Generator wants fake images to be classified as real (1)
            generator_label_decoded = self.discriminator(encoded_images) 
            label_real_for_generator = torch.ones_like(generator_label_decoded, device=self.device)
            generator_on_discriminator_loss = self.BCE_loss(generator_label_decoded, label_real_for_generator)
            # generator_on_discriminator_loss = self.BCE_loss(generator_label_decoded, self.label_encoded[:generator_label_decoded.size(0)])

            # edge loss compute 
            generator_loss_on_edge = self.combined_edge_loss(encoded_images, images)

            # MSE encoded_images and cover
            generator_loss_on_encoder = self.MSE_loss(encoded_images, images)

            # LPIPS loss for encoded_images and cover
            generator_loss_on_encoder_lpips = self.LPIPS_loss(encoded_images, images)
            generator_loss_on_encoder_ssim = 1 - self.SSIM_loss(encoded_images, images)

            # Output ,the decoded message should be the same as the input message
            generator_loss_on_decoder = self.BCE_loss(decoded_message, messages)

            # combine all losses           
            decoder_w, encoder_w, discriminator_w = self.ramp_function(epoch,65)

            generator_loss = discriminator_w * generator_on_discriminator_loss + \
                encoder_w * generator_loss_on_edge + \
                encoder_w * generator_loss_on_encoder + \
                encoder_w * generator_loss_on_encoder_lpips + \
                encoder_w * generator_loss_on_encoder_ssim + \
                decoder_w * generator_loss_on_decoder
            
            generator_loss.backward()
            self.optimizer_encoder_decoder.step()

            encoded_normalized = (encoded_images + 1.0) / 2.0
            images_normalized = (images + 1.0) / 2.0            
            psnr_loss = kornia.metrics.psnr(encoded_normalized, images_normalized, 1.0)
            #print("psnr_loss Encoder training and decoder training:",psnr_loss)

            # SSIM loss
            ssim_loss = kornia.metrics.ssim(encoded_images, images, window_size=5, max_val=2.0).mean()
            #print("ssim_loss Encoder training and decoder training:",ssim_loss)

            # ------------------------------------------------------------------------
            # Decoded message losses
            # ------------------------------------------------------------------------
            error_rate = self.decoded_message_error_rate_batch(messages, decoded_message)
            accuracy_rate = self.decoded_message_accuracy_batch(messages, decoded_message)

            losses_dict = {
                "error_rate": error_rate,
                "psnr_loss": psnr_loss,
                "ssim_loss": ssim_loss,
                "generator_loss": generator_loss,
                "generator_loss_on_discriminator": generator_on_discriminator_loss,
                "generator_loss_on_encoder": generator_loss_on_encoder,
                "generator_loss_on_decoder": generator_loss_on_decoder,
                "discriminator_cover_loss": disc_cover_loss,
                "discriminator_encoded_loss": disc_encoded_loss,
                "decode_accuracy" : accuracy_rate,
                "Lpips_loss_in_train" : generator_loss_on_encoder_lpips,
            }

            return losses_dict, messages, decoded_message
        
    def train_decoder(self, image: torch.Tensor, message: torch.Tensor, epoch: int):
        self.encoder_decoder.train()
        

        with torch.enable_grad():
            images, messages = image.to(self.device), message.to(self.device)

            encoded_images, noised_image, decoded_message = self.encoder_decoder(images, messages, epoch=epoch)

            # decoded_messages = self.encoder_decoder.module.decoder(noised_image)

            # ------------------------------------------------------------------------
            # Train decoder only
            # ------------------------------------------------------------------------
            self.optimizer_encoder_decoder.zero_grad()  
            generator_loss = self.BCE_loss(decoded_message, messages)

            generator_loss.backward()
            self.optimizer_encoder_decoder.step()  

            max_val = torch.max(encoded_images.max(), images.max())
            psnr_loss = kornia.metrics.psnr(encoded_images.detach(), images, max_val) 

            ssim_loss = 1 - 2 * kornia.metrics.ssim(encoded_images.detach(), images, window_size=5).mean()

        # ------------------------------------------------------------------------
        # Decoded message losses
        # ------------------------------------------------------------------------     
        error_rate = self.decoded_message_error_rate_batch(messages, decoded_message)
        accuracy_rate = self.decoded_message_accuracy_batch(messages, decoded_message)

        losses_dict = {
            "error_rate": error_rate,
            "psnr_loss": psnr_loss,
            "ssim_loss": ssim_loss,
            "generator_loss": generator_loss,
            "generator_loss_on_discriminator": 0.,
            "generator_loss_on_encoder": 0.,
            "generator_loss_on_decoder": 0.,
            "discriminator_cover_loss": 0.,
            "discriminator_encoded_loss": 0.,
            "decode_accuracy" : accuracy_rate,
        }

        return losses_dict, messages, decoded_message



    def validation(self, image: torch.Tensor, message: torch.Tensor, epoch: int):
        self.encoder_decoder.eval()
        self.discriminator.eval()

        # self.update_layers_epoch(self.encoder_decoder.module.noise.noise, epoch)

        with torch.no_grad():
            images, messages = image.to(self.device), message.to(self.device)
            encoded_images, noised_image, decoded_message = self.encoder_decoder(images, messages, epoch=epoch)
            # encoded_images = image + (encoded_images - image) * self.strength_factor

            # ------------------------------------------------------------------------
            # Validation Discriminator
            # ------------------------------------------------------------------------
            disc_label_cover = self.discriminator(images)
            label_cover = torch.ones_like(disc_label_cover, device=self.device)
            disc_cover_loss = self.BCE_loss(disc_label_cover, label_cover)


            disc_label_encoded = self.discriminator(encoded_images.detach())
            label_encoded = torch.zeros_like(disc_label_encoded, device=self.device)
            disc_encoded_loss = self.BCE_loss(disc_label_encoded, label_encoded)


            # ------------------------------------------------------------------------
            # Validation Encoder and Decoder
            # ------------------------------------------------------------------------            
            
            # Generator wants fake images to be classified as real (1)
            generator_label_decoded = self.discriminator(encoded_images) 
            label_real_for_generator = torch.ones_like(generator_label_decoded, device=self.device)
            generator_on_discriminator_loss = self.BCE_loss(generator_label_decoded, label_real_for_generator)

            # edge loss compute 
            generator_loss_on_edge = self.combined_edge_loss(encoded_images, images)
            # RAW : expected label for image should be "cover" : "1"
            generator_loss_on_encoder = self.MSE_loss(encoded_images, images)

            # LPIPS loss for encoded_images and cover
            generator_loss_on_encoder_lpips = self.LPIPS_loss(encoded_images, images)
            generator_loss_on_encoder_ssim = 1 - self.SSIM_loss(encoded_images, images)

            # Output ,the decoded message should be the same as the input message
            generator_loss_on_decoder = self.BCE_loss(decoded_message, messages)

            decoder_w, encoder_w, discriminator_w = self.ramp_function(epoch,65)
           
            generator_loss = discriminator_w * generator_on_discriminator_loss + \
                encoder_w * generator_loss_on_edge + \
                encoder_w * generator_loss_on_encoder + \
                encoder_w * generator_loss_on_encoder_lpips + \
                encoder_w * generator_loss_on_encoder_ssim + \
                decoder_w * generator_loss_on_decoder

            # PSNR loss
            encoded_normalized = (encoded_images + 1.0) / 2.0
            images_normalized = (images + 1.0) / 2.0            
            psnr_loss = kornia.metrics.psnr(encoded_normalized, images_normalized, 1.0)

            # SSIM loss
            ssim_loss = kornia.metrics.ssim(encoded_images, images, window_size=5, max_val=2.0).mean()

        # -------------------------------------------------------------------------
        # Decoded message losses
        # -------------------------------------------------------------------------
        error_rate = self.decoded_message_error_rate_batch(messages, decoded_message)
        accuracy_rate = self.decoded_message_accuracy_batch(messages, decoded_message)

        losses_dict = {
            "error_rate": error_rate,
            "psnr_loss": psnr_loss,
            "ssim_loss": ssim_loss,
            "generator_loss": generator_loss.item(),
            "generator_loss_on_discriminator": generator_on_discriminator_loss.item(),
            "generator_loss_on_encoder": generator_loss_on_encoder.item(),
            "generator_loss_on_decoder": generator_loss_on_decoder.item(),
            "discriminator_cover_loss": disc_cover_loss.item(),
            "discriminator_encoded_loss": disc_encoded_loss.item(),
            "decode_accuracy" : accuracy_rate,
            "Lpips_loss_in_train" : generator_loss_on_encoder_lpips,           
        }

        return losses_dict, (images, encoded_images, noised_image, messages, decoded_message)

    def save_trainable_parameters(self, path: str, include_gradients: bool = True, 
                                save_actual_params: bool = True, detailed_stats: bool = True,
                                save_rng_state: bool = True):
        """
        Save the trainable parameters of the model to a comprehensive text file.
        
        Args:
            path (str): Directory path to save the parameter information
            include_gradients (bool): Whether to include gradient statistics (default: False)
            save_actual_params (bool): Whether to save actual parameter values to a separate file (default: False)
            detailed_stats (bool): Whether to include detailed statistics per layer (default: True)
            save_rng_state (bool): Whether to save random number generator states (default: False)
        """
        import os
        import torch
        import json
        import random
        import numpy as np
        from datetime import datetime
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Collect trainable parameters with better error handling
        def get_module_params(module, module_name):
            try:
                if hasattr(module, 'module'):  # Handle DataParallel/DistributedDataParallel
                    actual_module = module.module
                else:
                    actual_module = module
                return {name: param for name, param in actual_module.named_parameters() if param.requires_grad}
            except AttributeError:
                print(f"Warning: Could not access parameters for {module_name}")
                return {}
        
        # Get parameters for each module
        encoder_params = get_module_params(self.encoder_decoder.module.encoder, "encoder") if hasattr(self.encoder_decoder, 'encoder') else \
                        get_module_params(self.encoder_decoder, "encoder_decoder")
        
        decoder_params = get_module_params(self.encoder_decoder.module.decoder, "decoder") if hasattr(self.encoder_decoder, 'decoder') else {}
        
        discriminator_params = get_module_params(self.discriminator, "discriminator") if hasattr(self, 'discriminator') else {}
        
        trainable_params = {
            "encoder": encoder_params,
            "decoder": decoder_params,
            "discriminator": discriminator_params
        }
        
        # Calculate parameter counts
        total_encoder_params = sum(p.numel() for p in encoder_params.values())
        total_decoder_params = sum(p.numel() for p in decoder_params.values())
        total_discriminator_params = sum(p.numel() for p in discriminator_params.values())
        total_all_params = total_encoder_params + total_decoder_params + total_discriminator_params
        
        total_trainable_params = {  # Fixed typo: "traiable" -> "trainable"
            "total_encoder_params": total_encoder_params,
            "total_decoder_params": total_decoder_params,
            "total_discriminator_params": total_discriminator_params,
            "total_params": total_all_params
        }
        
        # Calculate memory usage
        def calculate_memory_usage(params_dict):
            total_bytes = sum(p.numel() * p.element_size() for p in params_dict.values())
            return {
                'bytes': total_bytes,
                'kb': total_bytes / 1024,
                'mb': total_bytes / (1024 * 1024),
                'gb': total_bytes / (1024 * 1024 * 1024)
            }
        
        # Get parameter statistics
        def get_param_stats(param):
            with torch.no_grad():
                return {
                    'mean': param.data.mean().item(),
                    'std': param.data.std(unbiased=False).item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item(),
                    'norm': param.data.norm().item()
                }
        
        # Write comprehensive report
        report_file = os.path.join(path, "trainable_parameters_report.txt")
        with open(report_file, "w") as f:
            # Header
            f.write("="*80 + "\n")
            f.write("TRAINABLE PARAMETERS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"PyTorch version: {torch.__version__}\n")
            f.write(f"Device: {next(iter(encoder_params.values())).device if encoder_params else 'N/A'}\n")
            f.write("\n")
            
            # Summary statistics
            f.write("SUMMARY\n")
            f.write("-"*40 + "\n")
            for key, value in total_trainable_params.items():
                f.write(f"{key.replace('_', ' ').title()}: {value:,}\n")
            
            # Memory usage
            f.write(f"\nMemory Usage: {calculate_memory_usage(encoder_params)['mb']:.2f} MB "
                    f"(Encoder: {calculate_memory_usage(encoder_params)['mb']:.2f} MB, "
                    f"Decoder: {calculate_memory_usage(decoder_params)['mb']:.2f} MB, "
                    f"Discriminator: {calculate_memory_usage(discriminator_params)['mb']:.2f} MB)\n")
            
            f.write("\n" + "="*80 + "\n")
            
            # Detailed parameter information by module
            for module_name, params in trainable_params.items():
                if not params:
                    continue
                    
                f.write(f"\n{module_name.upper()} MODULE\n")
                f.write("-"*50 + "\n")
                f.write(f"Total parameters: {sum(p.numel() for p in params.values()):,}\n")
                f.write(f"Memory usage: {calculate_memory_usage(params)['mb']:.2f} MB\n\n")
                
                # Parameter details
                f.write("Layer Details:\n")
                for name, param in params.items():
                    f.write(f"  {name}:\n")
                    f.write(f"    Shape: {tuple(param.shape)}\n")
                    f.write(f"    Parameters: {param.numel():,}\n")
                    f.write(f"    Data type: {param.dtype}\n")
                    f.write(f"    Requires grad: {param.requires_grad}\n")
                    
                    if detailed_stats:
                        try:
                            stats = get_param_stats(param)
                            f.write(f"    Statistics: mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
                                f"norm={stats['norm']:.6f}\n")
                            f.write(f"    Range: [{stats['min']:.6f}, {stats['max']:.6f}]\n")
                        except Exception as e:
                            f.write(f"    Statistics: Error computing stats - {str(e)}\n")
                    
                    if include_gradients and param.grad is not None:
                        try:
                            grad_stats = get_param_stats(param.grad)
                            f.write(f"    Gradient stats: mean={grad_stats['mean']:.6f}, "
                                f"std={grad_stats['std']:.6f}, norm={grad_stats['norm']:.6f}\n")
                        except Exception as e:
                            f.write(f"    Gradient stats: Error - {str(e)}\n")
                    elif include_gradients:
                        f.write("    Gradient: None\n")
                    
                    f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        # Optionally save actual parameter values
        if save_actual_params:
            params_file = os.path.join(path, "parameter_values.pt")
            all_params = {}
            for module_name, params in trainable_params.items():
                if params:
                    all_params[module_name] = {name: param.data.clone() for name, param in params.items()}
            torch.save(all_params, params_file)
            print(f"Parameter values saved to: {params_file}")
        
        # Create a simple summary file as well (for backward compatibility)
        simple_file = os.path.join(path, "trainable_parameters.txt")
        with open(simple_file, "w") as f:
            f.write("Total Trainable Parameters:\n")
            for key, value in total_trainable_params.items():
                f.write(f"{key}: {value:,}\n")
            f.write("\nTrainable Parameters by Module:\n")
            for module_name, params in trainable_params.items():
                if params:
                    f.write(f"{module_name}:\n")
                    for name, param in params.items():
                        f.write(f"  {name}: {tuple(param.shape)} ({param.numel():,} params)\n")
        
        print(f"Parameter report saved to: {report_file}")
        print(f"Simple summary saved to: {simple_file}")
        print(f"Total trainable parameters: {total_all_params:,}")
        
        # Optionally save RNG states
        if save_rng_state:
            rng_file = os.path.join(path, "rng_state.json")
            rng_state = {
                'torch_rng_state': torch.get_rng_state().tolist(),
                'torch_cuda_rng_state': [],
                'numpy_rng_state': None,
                'python_rng_state': None,
                'initial_seed': torch.initial_seed()
            }
            
            # Save CUDA RNG states if available
            if torch.cuda.is_available():
                rng_state['torch_cuda_rng_state'] = [state.tolist() for state in torch.cuda.get_rng_state_all()]
            
            # Save NumPy RNG state if available
            try:
                np_state = np.random.get_state()
                rng_state['numpy_rng_state'] = {
                    'state': np_state[1].tolist(),
                    'pos': int(np_state[2]),
                    'has_gauss': int(np_state[3]),
                    'cached_gaussian': float(np_state[4])
                }
            except Exception:
                pass
            
            # Save Python random state
            try:
                rng_state['python_rng_state'] = random.getstate()
            except Exception:
                pass
            
            with open(rng_file, 'w') as f:
                json.dump(rng_state, f, indent=2)
            
            print(f"RNG state saved to: {rng_file}")


    # def save_trainable_parameters(self, path: str):
    #     """
    #     Save the trainable parameters of the model to a file.
    #     """

    #     # trainable_params = {name: param for name, param in self.encoder_decoder.named_parameters() if param.requires_grad}
    #     encoder_params = {name: param for name, param in self.encoder_decoder.module.encoder.named_parameters() if param.requires_grad}
    #     decoder_params = {name: param for name, param in self.encoder_decoder.module.decoder.named_parameters() if param.requires_grad}
    #     discriminator_params = {name: param for name, param in self.discriminator.module.named_parameters() if param.requires_grad}

    #     trainable_params = {
    #         "encoder": encoder_params,
    #         "decoder": decoder_params,
    #         "discriminator": discriminator_params
    #     }

    #     total_encoder_params = sum(p.numel() for p in self.encoder_decoder.module.encoder.parameters() if p.requires_grad)
    #     total_decoder_params = sum(p.numel() for p in self.encoder_decoder.module.decoder.parameters() if p.requires_grad)
    #     total_discriminator_params = sum(p.numel() for p in self.discriminator.module.parameters() if p.requires_grad)

    #     total_traiable_params = {
    #         "total_encoder_params": total_encoder_params,
    #         "total_decoder_params": total_decoder_params,
    #         "total_discriminator_params": total_discriminator_params,
    #         "total_params": total_encoder_params + total_decoder_params + total_discriminator_params
    #     }

    #     # Save the total parameters and trainable parameters to a file
    #     with open(os.path.join(path, "trainable_parameters.txt"), "w") as f:
    #         f.write("Total Trainable Parameters:\n")
    #         for key, value in total_traiable_params.items():
    #             f.write(f"{key}: {value}\n")
    #         f.write("\nTrainable Parameters by Module:\n")
    #         for module, params in trainable_params.items():
    #             f.write(f"{module}:\n")
    #             for name, param in params.items():
    #                 f.write(f"  {name}: {param.shape}\n")
                    

        # with open(path + "/trainable_parameters.txt", "w") as f:
        #     for module, params in trainable_params.items():
        #         f.write(f"{module}:\n")
        #         for name, param in params.items():
        #             f.write(f"  {name}: {param.shape}\n")


    def decoded_message_error_rate(self, message, decoded_message):
        length = message.shape[0]

        message = message.gt(0.5)
        decoded_message = decoded_message.gt(0.5)
        error_rate = float(sum(message != decoded_message)) / length
        return error_rate

    def decoded_message_error_rate_batch(self, messages, decoded_messages):
        error_rate = 0.0
        batch_size = len(messages)
        for i in range(batch_size):
            error_rate += self.decoded_message_error_rate(messages[i], decoded_messages[i])
        error_rate /= batch_size
        return error_rate

    def decoded_message_accuracy(self, message, decoded_message):
        length = message.shape[0]
        message = message.gt(0.5)
        decoded_message = decoded_message.gt(0.5)
        correct = float((message == decoded_message).sum())
        accuracy = correct / length
        return accuracy

    def decoded_message_accuracy_batch(self, messages, decoded_messages):
        accuracy = 0.0
        batch_size = len(messages)
        for i in range(batch_size):
            accuracy += self.decoded_message_accuracy(messages[i], decoded_messages[i])
        accuracy /= batch_size
        return accuracy

    def save_model(self, path_encoder_decoder: str, path_discriminator: str):
        torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
        torch.save(self.discriminator.module.state_dict(), path_discriminator)

    def load_model(self, path_encoder_decoder: str, path_discriminator: str):
        self.load_model_ed(path_encoder_decoder)
        self.load_model_dis(path_discriminator)

    def load_model_ed(self, path_encoder_decoder: str):
        self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder))

    def load_model_dis(self, path_discriminator: str):
        self.discriminator.module.load_state_dict(torch.load(path_discriminator))

    def calculate_BER_batch(self, messages, decoded_messages):
        total_bits = messages.numel()
        total_errors = (messages.gt(0.5) != decoded_messages.gt(0.5)).sum().item()
        ber = total_errors / total_bits
        return ber
    
    # def update_noise_layers(self, noise_layers, epoch):
    #     """
    #     Recursively update epoch-dependent noise layers.
    #     """
    #     for layer in noise_layers:
    #         if hasattr(layer, "set_epoch") and callable(getattr(layer, "set_epoch")):
    #             layer.set_epoch(epoch)
    #         elif isinstance(layer, Combined):
    #             self.update_noise_layers(layer.list, epoch)
    def update_layers_epoch(self, layers, epoch):
        for layer in layers:
            if hasattr(layer, "set_epoch") and callable(layer.set_epoch):
                layer.set_epoch(epoch)
            elif isinstance(layer, Combined):
                self.update_layers_epoch(layer.list, epoch)
