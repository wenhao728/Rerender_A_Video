#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/05/15 17:29:21
@Desc    :   Cleaned up batch inference template
@Ref     :   
'''
import os
import time
from pathlib import Path

import cv2
import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from PIL import Image
from tqdm import tqdm

from deps.ControlNet.annotator.canny import CannyDetector
from deps.ControlNet.annotator.util import HWC3
from deps.ControlNet.cldm.cldm import ControlLDM
from deps.ControlNet.cldm.model import create_model, load_state_dict
from deps.gmflow.gmflow.gmflow import GMFlow
from flow.flow_utils import get_warped_and_mask
from src.controller import AttentionControl
from src.ddim_v_hacked import DDIMVSampler
from src.img_util import find_flat_region, numpy2tensor


negative_prompt="ugly, blurry, low res, unaesthetic"

data_root = '/data/trc/videdit-benchmark/DynEdit'
method_name = 'model-name'

config = OmegaConf.create(dict(
    data_root=data_root,
    config_file=f'{data_root}/config.yaml',
    output_dir=f'{data_root}/outputs/{method_name}',
    # TODO define arguments
    model_ckpt='/data/trc/tmp-swh/models/ControlNet/models/control_sd15_canny.pth',
    vae_ckpt='/data/trc/tmp-swh/models/sd-vae-ft-mse-original/vae-ft-mse-840000-ema-pruned.ckpt',
    control_strength=0.7,
    num_samples=1,
    ddim_steps=20,
    scale=7.5,
    eta=0,
    seed=33,
    style_update_freq=10,
    x0_strength=0.05,  # -0.05, 0.25
    mask_period=(0.5, 0.8),
))


def load_flowmodel():
    flow_model = GMFlow(
        feature_channels=128,
        num_scales=1,
        upsample_factor=8,
        num_head=1,
        attention_type='swin',
        ffn_dim_expansion=4,
        num_transformer_layers=6,
    ).to('cuda')
    checkpoint = torch.load('models/gmflow_sintel-0c07dcb3.pth',
                            map_location=lambda storage, loc: storage)
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    flow_model.load_state_dict(weights, strict=False)
    flow_model.eval()
    return flow_model


@torch.no_grad()
def main():
    seed_everything(config.seed)
    # load model
    print('Loading models ...')
    device = torch.device('cuda')
    # TODO define model
    blur = T.GaussianBlur(kernel_size=(9, 9), sigma=(18, 18))
    canny_detector = CannyDetector()
    detector = lambda x: canny_detector(x, 50, 100)
    model: ControlLDM = create_model('./deps/ControlNet/models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(config.model_ckpt, location='cuda'))
    model = model.cuda()
    model.control_scales = [config.control_strength] * 13
    model.first_stage_model.load_state_dict(
        torch.load(config.vae_ckpt)['state_dict'], strict=False)
    # model.model.diffusion_model.forward = \
    #     freeu_forward(model.model.diffusion_model, *cfg.freeu_args)
    ddim_v_sampler = DDIMVSampler(model)

    flow_model = load_flowmodel()
    controller = AttentionControl(
        inner_strength=0.9, mask_period=(0.5, 0.8),
        cross_period=(0, 1), ada_period=(1, 1),
        warp_period=(0, 0.1), loose_cfattn=False)


    data_config = OmegaConf.load(config.config_file)
    preprocess_elapsed_ls = []
    inference_elapsed_ls = []
    for row in tqdm(data_config['data']):
        output_dir = Path(f"{config.output_dir}/{row.video_id}")
        if output_dir.exists():
            print(f"Skip {row.video_id} ...")
            continue
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        # load video
        print(f"Processing {row.video_id} ...")
        video_path = f'{config.data_root}/frames/{row.video_id}'
        # TODO load video
        imgs = sorted(os.listdir(video_path))
        imgs = [os.path.join(video_path, img) for img in imgs]

        # # Optional
        # inverse_path = Path(f"{config.output_dir}/{row.video_id}/.cache")
        # inverse_path.mkdir(parents=True, exist_ok=True)

        # preprocess
        start = time.perf_counter()
        # TODO preprocess video
        preprocess_elapsed = time.perf_counter() - start
        preprocess_elapsed_ls.append(preprocess_elapsed)

        # edit
        print(f'Editting {row.video_id} ...')
        start = time.perf_counter()
        for i, edit in tqdm(enumerate(row.edit)):
            # TODO edit
            # prompts=edit['prompt'],
            # negative_prompts=edit['src_words']+negative_prompt,
            # inversion_prompt=row['prompt'],
            # edit['tgt_words']
            output_frames = []

            frame = cv2.imread(imgs[0])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = HWC3(frame)
            H, W, C = img.shape
            img_ = numpy2tensor(img)
            encoder_posterior = model.encode_first_stage(img_.cuda())
            x0 = model.get_first_stage_encoding(encoder_posterior).detach()
            detected_map = detector(img)
            detected_map = HWC3(detected_map)
            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(1)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            cond = {
                'c_concat': [control],
                'c_crossattn':
                [model.get_learned_conditioning([edit['prompt']])]
            }
            un_cond = {
                'c_concat': [control],
                'c_crossattn':
                [model.get_learned_conditioning([edit['src_words']+negative_prompt])]
            }
            shape = (4, H // 8, W // 8)
            controller.set_task('initfirst')
            samples, _ = ddim_v_sampler.sample(
                config.ddim_steps, 1, shape, cond,
                verbose=False,
                eta=config.eta,
                unconditional_guidance_scale=config.scale,
                unconditional_conditioning=un_cond,
                controller=controller,
                x0=x0, strength=config.x0_strength)
            x_samples = model.decode_first_stage(samples)
            x_samples = (
                einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5
            ).cpu().numpy().clip(0, 255).astype(np.uint8)
            output_frames.append(Image.fromarray(x_samples[0]))

            pre_result = x_samples
            pre_img = img
            first_result = pre_result
            first_img = pre_img
            
            # the rest of frames
            for i in range(0, len(imgs) - 1):
                cid = i + 1
                frame = cv2.imread(imgs[cid])
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = HWC3(frame)
                img_ = numpy2tensor(img)
                encoder_posterior = model.encode_first_stage(img_.cuda())
                x0 = model.get_first_stage_encoding(encoder_posterior).detach()
                detected_map = detector(img)
                detected_map = HWC3(detected_map)
                control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
                control = torch.stack([control], dim=0)
                control = einops.rearrange(control, 'b h w c -> b c h w').clone()
                cond['c_concat'] = [control]
                un_cond['c_concat'] = [control]

                image1 = torch.from_numpy(pre_img).permute(2, 0, 1).float()
                image2 = torch.from_numpy(img).permute(2, 0, 1).float()
                warped_pre, bwd_occ_pre, bwd_flow_pre = get_warped_and_mask(
                    flow_model, image1, image2, pre_result, False)
                blend_mask_pre = blur(
                    F.max_pool2d(bwd_occ_pre, kernel_size=9, stride=1, padding=4))
                blend_mask_pre = torch.clamp(blend_mask_pre + bwd_occ_pre, 0, 1)

                image1 = torch.from_numpy(first_img).permute(2, 0, 1).float()
                warped_0, bwd_occ_0, bwd_flow_0 = get_warped_and_mask(
                    flow_model, image1, image2, first_result, False)
                blend_mask_0 = blur(
                    F.max_pool2d(bwd_occ_0, kernel_size=9, stride=1, padding=4))
                blend_mask_0 = torch.clamp(blend_mask_0 + bwd_occ_0, 0, 1)

                mask = 1 - F.max_pool2d(blend_mask_0, kernel_size=8)
                controller.set_warp(
                    F.interpolate(bwd_flow_0 / 8.0, scale_factor=1. / 8, mode='bilinear'), mask)

                controller.set_task('keepx0, keepstyle')
                samples, intermediates = ddim_v_sampler.sample(
                    config.ddim_steps,
                    1,
                    shape,
                    cond,
                    verbose=False,
                    eta=config.eta,
                    unconditional_guidance_scale=config.scale,
                    unconditional_conditioning=un_cond,
                    controller=controller,
                    x0=x0, strength=config.x0_strength)
                direct_result = model.decode_first_stage(samples)

                # pixel fusion
                blend_results = (1 - blend_mask_pre) * warped_pre + blend_mask_pre * direct_result
                blend_results = (1 - blend_mask_0) * warped_0 + blend_mask_0 * blend_results
                bwd_occ = 1 - torch.clamp(1 - bwd_occ_pre + 1 - bwd_occ_0, 0, 1)
                blend_mask = blur(F.max_pool2d(bwd_occ, kernel_size=9, stride=1, padding=4))
                blend_mask = 1 - torch.clamp(blend_mask + bwd_occ, 0, 1)

                encoder_posterior = model.encode_first_stage(blend_results)
                xtrg = model.get_first_stage_encoding(encoder_posterior).detach()  # * mask
                blend_results_rec = model.decode_first_stage(xtrg)
                encoder_posterior = model.encode_first_stage(blend_results_rec)
                xtrg_rec = model.get_first_stage_encoding(encoder_posterior).detach()
                xtrg_ = (xtrg + 1 * (xtrg - xtrg_rec))  # * mask
                blend_results_rec_new = model.decode_first_stage(xtrg_)
                tmp = (abs(blend_results_rec_new - blend_results).mean(dim=1, keepdims=True) > 0.25).float()
                mask_x = F.max_pool2d(
                    (F.interpolate(tmp, scale_factor=1 / 8., mode='bilinear') > 0).float(),
                    kernel_size=3, stride=1, padding=1)
                mask = (1 - F.max_pool2d(1 - blend_mask, kernel_size=8))  # * (1-mask_x)

                noise_rescale = find_flat_region(mask)
                masks = []
                for j in range(config.ddim_steps):
                    if (
                        j <= config.ddim_steps * config.mask_period[0] or 
                        j >= config.ddim_steps * config.mask_period[1]
                    ):
                        masks += [None]
                    else:
                        masks += [mask * config.mask_strength]
                
                xtrg = (xtrg + (1 - mask_x) * (xtrg - xtrg_rec)) * mask  # mask 1
                tasks = 'keepstyle, keepx0'
                if i % config.style_update_freq == 0:
                    tasks += ', updatestyle'
                controller.set_task(tasks, 1.0)
                samples, _ = ddim_v_sampler.sample(
                    config.ddim_steps,
                    config.num_samples,
                    shape,
                    cond,
                    verbose=False,
                    eta=config.eta,
                    unconditional_guidance_scale=config.scale,
                    unconditional_conditioning=un_cond,
                    controller=controller,
                    x0=x0, strength=config.x0_strength,
                    xtrg=xtrg,
                    mask=masks,
                    noise_rescale=noise_rescale)
                x_samples = model.decode_first_stage(samples)
                pre_result = x_samples
                pre_img = img
                viz = (
                    einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5
                ).cpu().numpy().clip(0, 255).astype(np.uint8)
                output_frames.append(Image.fromarray(viz[0]))
            
            output_frames[0].save(
                output_dir / f'{i}.gif', 
                save_all=True, append_images=output_frames[1:], optimize=False, 
                duration=83, loop=1,
            )

        inference_elapsed = time.perf_counter() - start
        inference_elapsed_ls.append(inference_elapsed)

    with open(f'{config.output_dir}/time.log', 'a') as f:
        f.write(f'Preprocess: {sum(preprocess_elapsed_ls)/len(preprocess_elapsed_ls):.2f} sec/video\n')
        n_prompts = len(row.edit)
        f.write(f'Edit:       {sum(inference_elapsed_ls)/len(inference_elapsed_ls)/n_prompts:.2f} sec/edit\n')
        f.write('Preprocess:\n')
        f.writelines([f'{e:.1f} ' for e in preprocess_elapsed_ls])
        f.write('\nEdit:\n')
        f.writelines([f'{e:.1f} ' for e in inference_elapsed_ls])
        f.write('\n')
    print('Everything done!')


if __name__ == '__main__':
    main()