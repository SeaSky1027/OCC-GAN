import os
import json
import logging
import argparse

import torch
from torch.utils.data import DataLoader

from audio_dataset import Audio_Dataset
from gan_model import Generator, Discriminator, count_parameters
from HiFiGanWrapper import HiFiGanWrapper
from FADWrapper import FADWrapper
from calculate_metric import calculate_metric
from train import train
from utils import save_json, load_json, seed_everything
from evaluation import evaluation

import tensorflow as tf
tf.get_logger().setLevel(logging.WARNING)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

id2class_dict: dict = {
    0:"dog_bark",
    1:"footstep",
    2:"gunshot",
    3:"keyboard",
    4:"moving_motor_vehicle",
    5:"rain",
    6:"sneeze_cough",
}

def main(args):
    seed_everything(args.seed)
    logging.info(f'Set Seed : {args.seed}')

    device = "cuda:{}".format(args.gpu)
    logging.info(f'Set GPU : {device}')
    logging.info('\n')

    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu], True)

    train_set = Audio_Dataset(data_path=args.train_data_path, id2class_dict=id2class_dict)
    train_loader = DataLoader(train_set, batch_size=args.batch, num_workers=8, shuffle=True)

    generator = Generator(noise_dim=args.noise_dim, embedding_dim=args.contrast_dim).to(device)
    count_parameters(generator, 'generator')

    discriminator = Discriminator(embedding_dim=args.contrast_dim).to(device)
    count_parameters(discriminator, 'discriminator')

    hifi_gan = HiFiGanWrapper(ckpt_path=args.ckpt_path)
    count_parameters(hifi_gan.hifi_gan_generator, 'hifi_gan')


    g_optim = torch.optim.Adam(generator.parameters(), lr=args.lr_G, betas=args.betas)
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=args.lr_D, betas=args.betas)

  
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.join('./tensorboard', args.name))

    evaluator = FADWrapper(ckpt_path=args.ckpt_path, id2class_dict=id2class_dict, number_of_audio=args.save_image_num, ground_truth_audio_samples_dir=args.test_data_path)
    score_dict = calculate_metric(args, generator, hifi_gan, evaluator, writer, 0, device)
    best_score = score_dict['average']

    result_dict = load_json(os.path.join("results/{}/".format(args.name), 'result.json'))
    result_dict['best_score']['epoch'] = 0
    result_dict['best_score']['FAD'] = score_dict
    save_json(result_dict, os.path.join("results/{}/".format(args.name), 'result.json'))

    state = {
        'epoch': 0,
        'iteration': 0,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'd_optim': d_optim.state_dict(),
        'g_optim': g_optim.state_dict(),
    }
    torch.save(state, os.path.join(args.ckpt_path, "{}.pt".format(args.name)))

    iteration = 0
    for epoch in range(args.epoch):
        epoch += 1
        logging.info(f'Epoch {epoch} Training.')
        d_loss, g_loss, iteration = train(
            args, epoch, train_loader, generator, discriminator, g_optim, d_optim, device, iteration, writer
        )
        logging.info('d_gan_loss {:.5f} / d_con_loss {:.5f} / d_total_loss {:.5f}'.format(d_loss['gan_loss'].avg, d_loss['contrastive_loss'].avg, d_loss['total_loss'].avg))
        logging.info('g_gan_loss {:.5f} / g_con_loss {:.5f} / g_total_loss {:.5f}'.format(g_loss['gan_loss'].avg, g_loss['contrastive_loss'].avg, g_loss['total_loss'].avg))
        logging.info('\n')

        if epoch % args.calc_metric_freq == 0:

            state = {
                'epoch': epoch,
                'iteration': iteration,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'd_optim': d_optim.state_dict(),
                'g_optim': g_optim.state_dict(),
            }
            torch.save(state, "checkpoint/{}/epoch_{}.pt".format(args.name, str(epoch).zfill(4)))

            score_dict = calculate_metric(args, generator, hifi_gan, evaluator, writer, epoch, device)

            if best_score >= score_dict['average']:
                logging.info('Best Metric Model Saving...')
                best_score = score_dict['average']
                torch.save(state, "checkpoint/{}/best_model.pt".format(args.name))
                torch.save(state, os.path.join(args.ckpt_path, "{}.pt".format(args.name)))

                result_dict = load_json(os.path.join("results/{}/".format(args.name), 'result.json'))
                result_dict['best_score']['epoch'] = epoch
                result_dict['best_score']['FAD'] = score_dict
                save_json(result_dict, os.path.join("results/{}/".format(args.name), 'result.json'))

            logging.info('Best Score : {:.3f}'.format(best_score))
            logging.info('\n')
    logging.info('End of training.')
    logging.info('\n')

    logging.info('Evaluation for best model ({} epoch).'.format(result_dict['best_score']['epoch']))
    logging.info('\n')

    result_dict = evaluation(args, evaluator, device)

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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--train_data_path", type=str, default="")
    parser.add_argument("--test_data_path", type=str, default="")
    parser.add_argument("--ckpt_path", type=str, default="")

    parser.add_argument("--name", default='test', type=str)

    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--batch", type=int, default=128)

    parser.add_argument("--lr_G", type=float, default=0.0004)
    parser.add_argument("--lr_D", type=float, default=0.0001)
    parser.add_argument("--betas", default=(0.5, 0.999))

    parser.add_argument('--noise_dim', type=int, default=128)
    parser.add_argument('--contrast_dim', default=128)

    parser.add_argument("--data_temp", type=float, default=0.1)
    parser.add_argument("--condition_temp", type=float, default=1.0)

    parser.add_argument("--conditional_contrastive_loss", action='store_true')
    parser.add_argument("--oneself_conditional_contrastive_loss", action='store_true')

    parser.add_argument('--calc_metric_freq', type=int, default=1)
    parser.add_argument("--save_image_num", type=int, default=100)
    parser.add_argument("--save_spectrogram", action='store_true')
    parser.add_argument("--n_candidate_gen_per_text", type=int, default=5)

    args = parser.parse_args()
    os.makedirs("checkpoint/{}/".format(args.name), exist_ok=True)
    os.makedirs("results/{}/".format(args.name), exist_ok=True)
    save_json({'best_score':{}}, os.path.join("results/{}/".format(args.name), 'result.json'))
    
    args_dict = vars(args)
    with open(os.path.join("results/{}/".format(args.name), 'argument.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    logging.info(args)
    logging.info('\n')

    file_handler = logging.FileHandler(os.path.join("results/{}/".format(args.name), 'Training.log'), mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    main(args)
