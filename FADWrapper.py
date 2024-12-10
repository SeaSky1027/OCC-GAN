from typing import Dict, List
import os
import argparse
import pickle
import logging

import numpy as np
import pandas as pd
import librosa

from create_embeddings_main import main
import fad_utils

class FADWrapper:
    def __init__(
        self,
        ckpt_path: str,
        id2class_dict: list,
        number_of_audio: int = 100,
        ground_truth_audio_samples_dir: str = "./DCASE_2023_Challenge_Task_7_Dataset/eval",
    ) -> None:
        self.vggish_ckpt_path = os.path.join(ckpt_path, 'vggish_model.ckpt')
        self.id2class_dict = id2class_dict
        self.number_of_audio = number_of_audio
        self.ground_truth_audio_samples_dir = ground_truth_audio_samples_dir
        self.output_dir: str = '/'.join(self.ground_truth_audio_samples_dir.split('/')[:-1] + ['eval_embedding'])
        os.makedirs(self.output_dir, exist_ok=True)

        self.sample_rate: int = 22050
        self.number_of_audio: int = number_of_audio
        self.audio_length_sec: float = 4.0

        logging.info("Extract embedding of ground truth audio.")
        for class_id, class_name in self.id2class_dict.items():
            self.extract_embedding('gt', self.ground_truth_audio_samples_dir, class_name)
        logging.info('\n')

    def extract_embedding(self, set_type, data_path, class_name):

        embeddig_output_dir = os.path.join(self.output_dir, class_name)
        files_path_list: list = [
            os.path.join(data_path, class_name, file_name)
            for file_name in os.listdir(
                os.path.join(data_path, class_name)
            )
            if os.path.splitext(file_name)[-1] == ".wav"
        ]

        self.sanity_check(files_path_list)
        self.write_meta_data(
            save_dir=embeddig_output_dir,
            save_file_name=f"{set_type}_files.cvs",
            file_path_list=files_path_list,
        )
        main(
            model_ckpt=self.vggish_ckpt_path,
            input_file_list_path=os.path.join(embeddig_output_dir, f'{set_type}_files.cvs'),
            output_path=os.path.join(embeddig_output_dir, set_type)
        )

    def compute_fad(self, generated_audio_samples_dir):
        logging.info("Extract embedding of generated audio & Calculate FAD score.")
        result_dict: dict = {"Category": list(), "FAD": list()}
        score_dict = {}
        for class_id, class_name in self.id2class_dict.items():
            mu_sigma_dict: dict = {'gt':{"mu": 0, "sigma": 0}, 'generated':{"mu": 0, "sigma": 0}}

            for set_type in mu_sigma_dict:
                if set_type == 'generated':
                    self.extract_embedding(set_type, generated_audio_samples_dir, class_name)
                embeddig_output_dir = os.path.join(self.output_dir, class_name)
                
                with open(
                    os.path.join(embeddig_output_dir, f'{set_type}_embedding.pkl'), "rb"
                ) as pickle_file:
                    data = pickle.load(pickle_file)
                mu_sigma_dict[set_type]["mu"] = np.array(data["mu"].float_list.value)
                emb_len = np.array(data["embedding_length"].int64_list.value)[0]
                mu_sigma_dict[set_type]["sigma"] = (
                    np.array(data["sigma"].float_list.value)
                ).reshape((emb_len, emb_len))

            fad = fad_utils.frechet_distance(
                mu_sigma_dict["gt"]["mu"],
                mu_sigma_dict["gt"]["sigma"],
                mu_sigma_dict["generated"]["mu"],
                mu_sigma_dict["generated"]["sigma"],
            )
            fad = round(fad, 3)

            result_dict["Category"].append(class_name)
            result_dict["FAD"].append(fad)
            score_dict[class_name] = fad
            pd.DataFrame(result_dict).to_csv(f"{self.output_dir}/fad.csv", index=False)

        average = round(sum(result_dict["FAD"]) / len(result_dict["FAD"]), 3)
        result_dict["Category"].append('average')
        result_dict["FAD"].append(average)
        score_dict['average'] = average
        pd.DataFrame(result_dict).to_csv(os.path.join(self.output_dir, 'fad.csv'), index=False)
        return score_dict

    def sanity_check(self, file_path_list: List[str]) -> None:
        assert (
            len(file_path_list) >= self.number_of_audio
        ), f"[Error]The number of audio is {len(file_path_list)}. there should be same or bigger {self.number_of_audio} wav files in each subfolder."
        for file_path in file_path_list:
            audio_data, sr = librosa.load(file_path, sr=None, mono=False)
            assert (
                sr == self.sample_rate
            ), f"sample rate should be {self.sample_rate}, but found {sr} at {file_path}."
            assert (
                audio_data.ndim == 1
            ), f"audio should be mono, but this file seems not: {file_path}"
            assert (
                len(audio_data) == int(self.sample_rate * self.audio_length_sec)
            ), f"length is expected to be 88200, but found {len(audio_data)}."

    def write_meta_data(
        self, save_dir: str, save_file_name: str, file_path_list: List[str]
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/{save_file_name}", "w") as file:
            for i, file_path in enumerate(file_path_list):
                if i != 0:
                    file.write("\n")
                file.write(file_path)
                if i + 1 == self.number_of_audio:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, default="./DCASE_2023_Challenge_Task_7_Dataset/eval")
    parser.add_argument("--number_of_audio", type=int, default=100)
    args = parser.parse_args()

    id2class_dict: dict = {
        0:"dog_bark",
        1:"footstep",
        2:"gunshot",
        3:"keyboard",
        4:"moving_motor_vehicle",
        5:"rain",
        6:"sneeze_cough",
    }

    evaluator = FADWrapper(id2class_dict=id2class_dict, number_of_audio=args.number_of_audio, ground_truth_audio_samples_dir=args.test_data_path)
    score_dict = evaluator.compute_fad('./synthesized_data/test/epoch_0')
    logging.info(score_dict)