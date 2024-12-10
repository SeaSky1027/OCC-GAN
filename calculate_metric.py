import os
import logging
import shutil
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

import torch

from utils import save_json, load_json

def save_spectrogram(fake_mel, save_path):
    fig, axs = plt.subplots(1, 1)
    axs.set_title('Mel Spectrogram (db)')
    axs.set_ylabel('freq_bin')
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.amplitude_to_db(fake_mel), origin='lower', aspect='auto')
    axs.set_xlim((0, 400))
    plt.savefig(save_path)

def calculate_metric(args, generator, vocoder, evaluator, writer, epoch, device='cuda:0'):
    epoch_name = 'epoch_{}'.format(epoch)

    os.makedirs('./synthesized_data/{}'.format(args.name), exist_ok=True)
    os.makedirs('./synthesized_data/{}/{}'.format(args.name, epoch_name), exist_ok=True)
    save_dir = './synthesized_data/{}/{}'.format(args.name, epoch_name)

    for class_id, class_name in evaluator.id2class_dict.items():
        os.makedirs(os.path.join(save_dir, class_name), exist_ok=True)

    logging.info('Generate samples.')
    generator.eval()
    with torch.no_grad():
        for label in range(7):
            class_name = evaluator.id2class_dict[label]
            logging.info(f'Generate {evaluator.number_of_audio} samples for {class_name} class.')

            labels = torch.LongTensor([label]).expand(evaluator.number_of_audio).to(device)
            fake_mel = generator(labels)

            for num in range(evaluator.number_of_audio):
                fake_wav = vocoder.generate_audio(fake_mel[num])
                fake_wav = np.concatenate((fake_wav, fake_wav[-136:]), axis=0)

                file_name = str(num+1).zfill(3)
                sf.write(os.path.join(save_dir, class_name, f'{file_name}.wav'),
                    fake_wav,
                    samplerate=22050,
                )

                if args.save_spectrogram:
                    save_spectrogram(fake_mel[num].squeeze().detach().cpu(), os.path.join(save_dir, class_name, f'{file_name}.png'))
    logging.info('\n')
    score_dict = evaluator.compute_fad(save_dir)

    logging.info('\n')
    logging.info('*'*30)
    logging.info('Epoch {} Results'.format(epoch))
    logging.info('*'*30)
    for class_name, fad_score in score_dict.items():
        writer.add_scalar('FAD/{}'.format(class_name), fad_score, epoch)
        if class_name == 'average':
            logging.info('*'*30)
        logging.info('{} : {}'.format(class_name, fad_score))
    logging.info('*'*30)
    logging.info('\n')
    shutil.move(os.path.join(evaluator.output_dir, 'fad.csv'), os.path.join(save_dir, 'fad.csv'))

    result_dict = load_json(os.path.join("results/{}/".format(args.name), 'result.json'))
    result_dict[epoch_name] = score_dict
    save_json(result_dict, os.path.join("results/{}/".format(args.name), 'result.json'))

    return score_dict
