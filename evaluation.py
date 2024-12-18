import os
import time
import logging
import argparse
from tqdm import tqdm
from collections import defaultdict

import torch
import torchaudio

from FADWrapper import FADWrapper
from synthesizer import Synthesizer
from utils import save_json, load_json

def evaluation(args, evaluator, id2class_dict, device='cuda:0'):

    synthesizer = Synthesizer(args.name, args.ckpt_path, id2class_dict, device=device)

    save_dir = './synthesized_data/evlauation'
    os.makedirs(save_dir, exist_ok=True)

    for class_id, class_name in id2class_dict.items():
        os.makedirs(os.path.join(save_dir, class_name), exist_ok=True)

    clap_score_dict = defaultdict(int)
    logging.info(f'Generates {args.n_candidate_gen_per_text} samples per data, of which the highest CLAP score is selected.')
    logging.info('\n')
    with torch.no_grad():
        for label in range(len(id2class_dict)):
            logging.info(f'Generate {evaluator.number_of_audio} samples for {class_name} class.')
            class_name = id2class_dict[label]

            for num in tqdm(range(evaluator.number_of_audio)):
                similarity, fake_sound, _ = synthesizer.generate_audio(label, n_candidate_gen_per_text=args.n_candidate_gen_per_text)
                clap_score_dict[class_name] += similarity

                file_name = str(num+1).zfill(3)
                torchaudio.save(uri=os.path.join(save_dir, class_name, f'{file_name}.wav'),
                                src=fake_sound,
                                sample_rate=22050)

            clap_score_dict[class_name] = round(clap_score_dict[class_name] / evaluator.number_of_audio, 3)

    logging.info('\n')
    clap_score_dict['average'] = round(sum([score for score in clap_score_dict.values()]) / len(id2class_dict), 3)
    fad_score_dict = evaluator.compute_fad(save_dir)

    result_dict = {'FAD':fad_score_dict, 'CLAP':clap_score_dict}

    save_json(result_dict, os.path.join("results/{}/".format(args.name), 'evaluation.json'))

    return result_dict
    
def main(args):
    device = "cuda:{}".format(args.gpu)
        
    id2class_dict: dict = {
        0:"dog_bark",
        1:"footstep",
        2:"gunshot",
        3:"keyboard",
        4:"moving_motor_vehicle",
        5:"rain",
        6:"sneeze_cough",
    }

    evaluator = FADWrapper(ckpt_path=args.ckpt_path, id2class_dict=id2class_dict, number_of_audio=args.save_image_num, ground_truth_audio_samples_dir=args.test_data_path)

    result_dict = evaluation(args, evaluator, id2class_dict, device)

    for score_name, score_dict in result_dict.items():
        logging.info('\n')
        logging.info('*'*30)
        logging.info(f'{score_name} Score')
        logging.info('*'*30)
        for class_name, score in score_dict.items():
            if class_name == 'average':
                logging.info('*'*30)
            logging.info('{} : {}'.format(class_name, score))
        logging.info('*'*30)
    logging.info('\n')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--name", type=str, default="generator")

    parser.add_argument("--save_image_num", type=int, default=100)
    parser.add_argument("--n_candidate_gen_per_text", type=int, default=5)

    parser.add_argument("--test_data_path", type=str, default="./DCASE_2023_Challenge_Task_7_Dataset/eval")
    parser.add_argument("--ckpt_path", type=str, default="./pretrained_checkpoints/")

    args = parser.parse_args()

    main(args)
