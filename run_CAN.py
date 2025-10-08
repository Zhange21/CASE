import os.path
import sys
import json
import argparse
import math

import os
import matplotlib.pyplot as plt

sys.path.append('..')
sys.path.append('/root/zzg/easyedit0427/EasyEdit-main')
from easyeditor import (
    FTHyperParams,
    GraceHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    MENDHyperParams,
    WISEHyperParams,
    BaseEditor,
    summary_metrics,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', type=str)
    parser.add_argument('--hparams_dir', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--data_type', type=str,
                        choices=['ZsRE', 'counterfact', 'temporal', 'hallucination'])
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--ds_size', default=3, type=int)
    parser.add_argument('--device', default=-1, type=int)
    parser.add_argument('--sequential_edit', action="store_true")
    parser.add_argument('--method_flag', default='cluster', type=str)

    parser.add_argument('--plt_dir', default='/data/zzg/Edit/plot_figure', type=str)
    # parser.add_argument('--maxsimseq', default=False, type=bool)
    parser.add_argument('--begin_data_index', default=0, type=int)
    parser.add_argument('--data_json', default='', type=str)


    args = parser.parse_args()
    print(args, flush=True)

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'WISE' or args.editing_method == 'GRADMOE':
        editing_hparams = WISEHyperParams
    elif args.editing_method == 'CAN':
        editing_hparams = WISEHyperParams
    else:
        raise NotImplementedError

    K = args.ds_size
    K_0 = args.begin_data_index

    if args.data_type == 'ZsRE':
        # all_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_eval.json', 'r', encoding='utf-8'))[:500]
        # json_results = json.dumps(all_data, ensure_ascii=False, indent=2)
        # with open('/data/zzg/Edit/test_out/data_debug.json', 'w', encoding='utf-8') as f:eval_random_1000
        #     f.write(json_results)
        if args.data_json == '':
            edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_eval_random_1000.json', 'r', encoding='utf-8'))[K_0:K_0+K]
        else:
            edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/{args.data_json}.json', 'r', encoding='utf-8'))[K_0:K_0+K]
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_train.json', 'r', encoding='utf-8'))[K_0:K_0+K]
        loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]

        prompts = [edit_data_['src'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        # subject = None
        rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in edit_data]
        target_new = [edit_data_['alt'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood':{
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }
    elif args.data_type == 'hallucination':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/hallucination-edit.json', 'r', encoding='utf-8'))[K_0:K_0+K]
        # loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/hallucination-train.json', 'r', encoding='utf-8'))[K_0:K_0+K]

        K_0l = math.floor(K_0 / 2.5)
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/hallucination-train.json', 'r', encoding='utf-8'))[K_0l:K_0l+K]

        loc_prompts = [edit_data_['locality_prompt'] + ' ' + edit_data_['locality_ground_truth'] for edit_data_ in loc_data]

        prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        if args.editing_method == 'CAN':
            subject = None
        rephrase_prompts = None
        target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood': {
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }
    elif args.data_type == 'counterfact':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/counterfact-edit.json', 'r', encoding='utf-8'))[K_0:K_0+K]
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/counterfact-edit.json', 'r', encoding='utf-8'))[K_0:K_0+K]
        loc_prompts = [edit_data_['locality_prompt'] + ' ' + edit_data_['locality_ground_truth'] for edit_data_ in loc_data]

        prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        rephrase_prompts = [edit_data_['rephrase_prompt'] for edit_data_ in edit_data]
        target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood':{
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }

    hparams = editing_hparams.from_hparams(f'{args.hparams_dir}')
    if args.device >= 0:
        hparams.device = args.device

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f'{hparams.model_name.split("/")[-1]}_{args.editing_method}_N={args.ds_size}_Sequential={args.sequential_edit}_method={args.method_flag}.json'
        )

    print("See results at: ", output_file)

    editor = BaseEditor.from_hparams(hparams)

    if args.editing_method == 'WISE':
        metrics, edited_model = editor.edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            target_new=target_new,
            loc_prompts=loc_prompts,
            subject=subject,
            locality_inputs=locality_inputs,
            sequential_edit=args.sequential_edit,
            eval_metric='ppl' if args.data_type == 'hallucination' else 'token em'
        )
    else:
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            target_new=target_new,
            loc_prompts=loc_prompts,
            subject=subject,
            locality_inputs=locality_inputs,
            sequential_edit=args.sequential_edit,
            eval_metric='ppl' if args.data_type == 'hallucination' else 'token em'
        )

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    # if 'pre' in metrics[0] and 'post' in metrics[0]:

    #     sample_indices = range(len(metrics))
    #     plt_folder = f'{args.plt_dir}/{hparams.model_name.split("/")[-1]}_{args.editing_method}_N={args.ds_size}_Sequential={args.sequential_edit}_method={args.method_flag}'
    #     os.makedirs(plt_folder, exist_ok=True)
    #     all_edit_flag = ['pre','post']
    #     for edit_flag in all_edit_flag:
    #         # all_metric_name = metrics[0][edit_flag].keys()
    #         all_metric_name = ['rewrite_acc', 'rephrase_acc']
    #         for metric_name in all_metric_name:
    #             # if len(metrics[0][edit_flag][metric_name]) !=0:
    #             values = [sample[edit_flag][metric_name][0] for sample in metrics]
    #             if sorted_idx is not None:
    #                 values = [values[i] for i in sorted_idx]
    #             # post_values = [sample['post'][metric_name] for sample in metrics]

    #             plt.figure(figsize=(10, 6))
    #             bars = plt.bar(sample_indices, values, color='skyblue')

    #             for bar in bars:
    #                 height = bar.get_height()
    #                 plt.text(bar.get_x() + bar.get_width()/2., height,
    #                         f'{height:.2f}',
    #                         ha='center', va='bottom', fontsize=6)

    #             plt.title(f' {metric_name}_{edit_flag} ', fontsize=14)
    #             plt.xlabel('samples', fontsize=12)
    #             plt.ylabel(f'{metric_name.upper()}_{edit_flag}', fontsize=12)
    #             if sorted_idx is not None:
    #                 plt.xticks(sample_indices, sorted_idx)  # 显示所有样本编号
    #             else:
    #                 plt.xticks(sample_indices)
    #             plt.ylim(0, 1.05)  # 根据指标范围调整（例如分类指标通常0-1）
    #             plt.grid(axis='y', linestyle='--', alpha=0.7)

    #             plt.tight_layout()

    #             plt.savefig(f'{plt_folder}/{edit_flag}_{metric_name}_performance.png')
    #             plt.close()

    if len(metrics) > 0:
        summary_metrics(metrics)

