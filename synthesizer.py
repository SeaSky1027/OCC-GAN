import os
import logging
import numpy as np

import torch
import torchaudio

import laion_clap as CLAP
from gan_model import Generator, count_parameters
from HiFiGanWrapper import HiFiGanWrapper

class Synthesizer:
    def __init__(self, model_name, ckpt_path, id2class_dict, device='cpu'):
        self.device = device
        self.id2class_dict = id2class_dict

        self.generator = Generator().to(device)
        ckpt_file = "{}.pt".format(model_name)
        ckpt = torch.load(os.path.join(ckpt_path, ckpt_file), map_location=device)

        logging.info('Generator Setting')
        self.generator.load_state_dict(ckpt["generator"])
        self.generator.eval()
        num_params = count_parameters(self.generator)
        logging.info(f"Number of generator parameters: {num_params / 1000000:.2f} M")
        logging.info('\n')

        logging.info('Vocoder Setting')
        self.vocoder = HiFiGanWrapper(ckpt_path=ckpt_path)
        num_params = count_parameters(self.vocoder.generator)
        logging.info(f"Number of vocoder parameters: {num_params / 1000000:.2f} M")
        logging.info('\n')

        self.test_clap_model = CLAP.CLAP_Module(enable_fusion=False, device=device)
        self.test_clap_model.load_ckpt(os.path.join(ckpt_path, '630k-audioset-best.pt'))
        self.test_clap_model.eval()
        logging.info('CLAP Setting.')
        logging.info('\n')

    @torch.no_grad()
    def generate_audio(self, class_id, n_candidate_gen_per_text=5):
        class_name = self.id2class_dict[class_id]

        text_prompt = ' '.join(class_name.split('_'))
        if text_prompt[0].islower():
            text_prompt = text_prompt[0].upper() + text_prompt[1:]

        lables = [class_id] * n_candidate_gen_per_text
        lables = torch.LongTensor(lables).to(self.device)
        
        fake_mel = self.generator(lables)
        fake_mel = fake_mel.squeeze()
        fake_sound_list = []
        for num in range(n_candidate_gen_per_text):
            fake_sound = self.vocoder.generate_audio(fake_mel[num])
            fake_sound = np.concatenate((fake_sound, fake_sound[-136:]), axis=0)
            fake_sound_list.append(torch.from_numpy(fake_sound))
            
        fake_sound = torch.stack(fake_sound_list, dim=0)

        text_prompt = [text_prompt] * n_candidate_gen_per_text
        
        fake_sound_48k = torchaudio.functional.resample(fake_sound, orig_freq=22050, new_freq=48000)
        audio_embeddings = self.test_clap_model.get_audio_embedding_from_data(fake_sound_48k.squeeze(), use_tensor=True)
        _, text_embeddings = self.test_clap_model.get_text_embedding(text_prompt, use_tensor=True)
        similarity = torch.nn.functional.cosine_similarity(audio_embeddings, text_embeddings, dim=-1, eps=1e-8)

        best_index = torch.argmax(similarity).item()
        best_similarity = similarity[best_index]
        best_sound = fake_sound[best_index].unsqueeze(0)
        
        best_sound = best_sound.detach().cpu()

        best_similarity = round(best_similarity.item(), 3)
        
        return best_similarity, best_sound, class_name
