import os
import time
import argparse

import torch
import torchaudio

from synthesizer import Synthesizer

id2class_dict: dict = {
    0:"dog_bark",
    1:"footstep",
    2:"gunshot",
    3:"keyboard",
    4:"moving_motor_vehicle",
    5:"rain",
    6:"sneeze_cough",
}

def to_MB(a):
    return a/1024.0/1024.0
def to_GB(a):
    return a/1024.0/1024.0/1024.0
    
def main(args):
    device = "cuda:{}".format(args.gpu)
    
    before_model_to_device = torch.cuda.memory_allocated(device=device)
    print(f"Before model to device: {to_MB(before_model_to_device):.2f}MB")

    synthesizer = Synthesizer(args.name, args.ckpt_path, id2class_dict, device=device)

    after_model_to_device = torch.cuda.memory_allocated(device=device)
    print(f"After model to device: {to_MB(after_model_to_device):.2f}MB")
    print()

    os.makedirs(args.save_path, exist_ok=True)

    with torch.no_grad():
        print(f'Create Audio for {id2class_dict[args.class_id]}')

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)   # Time evaluation 
        starter.record()
        similarity, fake_sound, class_name = synthesizer.generate_audio(args.class_id, n_candidate_gen_per_text=args.n_candidate_gen_per_text)
        ender.record()

        print(f'Best similarity among generated {args.n_candidate_gen_per_text} sounds : {similarity}')
        print()

        after_model_inference = torch.cuda.memory_allocated(device=device)
        print(f"After model inference: {to_MB(after_model_inference):.2f}MB")

        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)

        torchaudio.save(uri=os.path.join(args.save_path, f'{class_name}.wav'),
                        src=fake_sound,
                        sample_rate=22050)

        print(f'Inference Time : {round(curr_time * 1e-3, 3)}s')
        print()

    print(f"Before model to device: {to_MB(before_model_to_device):.2f}MB = {to_GB(before_model_to_device):.2f}GB")
    print(f"After model to device: {to_MB(after_model_to_device):.2f}MB = {to_GB(after_model_to_device):.2f}GB")
    print(f"After model inference: {to_MB(after_model_inference):.2f}MB = {to_GB(after_model_inference):.2f}GB")
    print()

    num_params = sum(p.numel() for p in synthesizer.generator.parameters())
    print(f"Number of Generator parameters: {num_params / 1000000:.2f} M")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--name", type=str, default="generator")

    parser.add_argument("--class_id", type=int, default=0)
    parser.add_argument("--n_candidate_gen_per_text", type=int, default=5)

    parser.add_argument("--save_path", type=str, default="./inference_results")
    parser.add_argument("--ckpt_path", type=str, default="./pretrained_checkpoints/")

    args = parser.parse_args()

    main(args)
