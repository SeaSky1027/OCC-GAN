import os
import librosa
import logging
from collections import defaultdict
from scipy.io.wavfile import read as loadwav

import torch

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

MAX_WAV_VALUE = 32768.0

mel_basis = {}
hann_window = {}

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def mel_spectrogram_hifi(
    audio, n_fft, n_mels, sample_rate, hop_length, fmin, fmax, center=False
):
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)

    if torch.min(audio) < -1.0:
        logging.info('min value is ', torch.min(audio))
    if torch.max(audio) > 1.0:
        logging.info('max value is ', torch.max(audio))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel_fb = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str(fmax) + '_' + str(audio.device)] = (
            torch.from_numpy(mel_fb).float().to(audio.device)
        )
        hann_window[str(audio.device)] = torch.hann_window(n_fft).to(audio.device)
    audio = torch.nn.functional.pad(
        audio.unsqueeze(1),
        (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
        mode='reflect',
    )
    audio = audio.squeeze(1)
    
    spec = torch.stft(
        audio,
        n_fft,
        hop_length=hop_length,
        window=hann_window[str(audio.device)],
        center=center,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)

    mel = torch.matmul(mel_basis[str(fmax) + '_' + str(audio.device)], spec)
    mel = spectral_normalize_torch(mel).numpy()

    return mel

class Audio_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        id2class_dict,
        max_length = 22050 * 4,
        n_fft = 1024,
        n_mels = 80,
        hop_length = 256,
        sample_rate = 22050,
        fmin = 0,
        fmax = 8000,
    ):
        self.data_path = data_path
        self.id2class_dict = id2class_dict
        self.max_length = max_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.fmin = fmin
        self.fmax = fmax

        self.load_datalist()

    def sanity_check(self):
        dir_list = os.listdir(self.data_path)

        for class_name in self.id2class_dict.values():
            assert (class_name in dir_list), f"[Error] Folder {class_name} does not exist."
            assert (len(os.path.join(self.data_path, class_name)) == 0), f"[Error] Folder {class_name} is empty."

    def load_datalist(self):
        class2id_dict = {value: key for key, value in self.id2class_dict.items()}
        
        data_num_dict = defaultdict(int)
        self.data_list = []
        for root_dir, _, file_list in os.walk(self.data_path):
            for file_name in file_list:
                if os.path.splitext(file_name)[-1] == ".wav":
                    class_name = root_dir.split("/")[-1]

                    self.data_list.append(
                        {
                            "file_path": f"{root_dir}/{file_name}",
                            "class_id": class2id_dict[class_name]
                        }
                    )

                    data_num_dict[class_name] += 1

        logging.info('Data Info')
        for class_id, class_name in self.id2class_dict.items():
            logging.info('class_id : {} | data_num : {} | class_name : {}'.format(
                class_id,
                data_num_dict[class_name],
                class_name,
            ))
        logging.info('\n')

    def __getitem__(self, index):
        filename = self.data_list[index]['file_path']
        class_id = self.data_list[index]['class_id']

        sample_rate, audio = loadwav(filename)
        audio = audio / MAX_WAV_VALUE

        if sample_rate != self.sample_rate:
            raise ValueError(
                "{} sr doesn't match {} sr ".format(sample_rate, self.sample_rate)
            )
        
        if len(audio) > self.max_length:
            audio = audio[0 : self.max_length]

        if len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)), 'constant')

        mel_spec = mel_spectrogram_hifi(
            audio,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        return mel_spec, class_id

    def __len__(self):
        return len(self.data_list)

if __name__ == '__main__':

    id2class_dict: dict = {
        0:"dog_bark",
        1:"footstep",
        2:"gunshot",
        3:"keyboard",
        4:"moving_motor_vehicle",
        5:"rain",
        6:"sneeze_cough",
    }

    from torch.utils.data import DataLoader
    train_set = Audio_Dataset(data_path="./DCASE_2023_Challenge_Task_7_Dataset/dev", id2class_dict=id2class_dict)
    train_loader = DataLoader(train_set, batch_size=4, num_workers=8, shuffle=True)

    for mel_spec, class_id in train_loader:
        print('mel_spec :', mel_spec.shape)
        print('class_id :', class_id)
        break
