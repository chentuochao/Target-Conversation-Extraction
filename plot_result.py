import pandas as pd
import argparse
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import scipy.stats

def main(args: argparse.Namespace):
    
    results_csv_path = args.results_dir
    results_df = pd.read_csv(results_csv_path)

    mean_single_sisdri = np.mean(results_df['sisdri_target'])
    std_single_sisdri = np.std(results_df['sisdri_target'])
    print(f'Single target SI-SDRi: {mean_single_sisdri:.03f} +/- {std_single_sisdri:.03f}dB')
    
    mean_single_input_sisdr = np.mean(results_df['input_sisdr_target'])
    std_single_input_sisdr = np.std(results_df['input_sisdr_target'])
    print(f'Single target input SI-SDR: {mean_single_input_sisdr:.03f} +/- {std_single_input_sisdr:.03f}dB')


    mean_single_sisdri = np.mean(results_df['snri'])
    std_single_sisdri = np.std(results_df['snri'])
    print(f'Single target SNRi: {mean_single_sisdri:.03f} +/- {std_single_sisdri:.03f}dB')
    
    mean_single_input_sisdr = np.mean(results_df['input_snr'])
    std_single_input_sisdr = np.std(results_df['input_snr'])
    print(f'Single target input SNR: {mean_single_input_sisdr:.03f} +/- {std_single_input_sisdr:.03f}dB')


    overlap_ratio = np.mean(results_df['overlap_ratio'])
    print(f'overlap_ratio: {overlap_ratio:.05f}')
    
    overlap_ratio_inter = np.mean(results_df['overlap_ratio_inter'])
    print(f'overlap_ratio_inter: {overlap_ratio_inter:.05f}')

    mean_single_sisdri = np.mean(results_df['snri'])
    std_single_sisdri = np.std(results_df['snri'])
    print(f'Single target SNRi: {mean_single_sisdri:.03f} +/- {std_single_sisdri:.03f}dB')

    wrong_ratio =  (np.sum(results_df['snri_interfer'] > results_df['snri_other'])  )/len(results_df['snri_other'])
    print("wrong ratio = ", wrong_ratio)
    
    print(np.sum(results_df['sisdri_target'] < 0), len(results_df['sisdri_target']))

    if 1:
        plt.figure(figsize=(4, 3))
        plt.scatter(results_df['input_sisdr_target'], results_df['input_sisdr_target'] + results_df['sisdri_target'], s=3)
        plt.plot([min(results_df['input_sisdr_target']), max(results_df['input_sisdr_target'])], 
                [min(results_df['input_sisdr_target']), max(results_df['input_sisdr_target'])], color='black', linestyle = "--")
        plt.xlabel('Input SI-SDR (dB)', fontsize = 14)
        plt.ylabel('Output SI-SDR (dB)', fontsize = 14)
        plt.ylim([-17, 20])
        plt.tight_layout()
        plt.savefig( os.path.join('./debug/input_vs_output_si_sdr.png') )
        plt.clf()



        duration_x = [[0, 15], [15, 25], [25, 35], [35, 45], [45, 60]]
        duration_y = [[] for i in range(len(duration_x))]
        for i in range(len(results_df['self_duration'])):
            d = results_df['self_duration'][i]
            sisdri = results_df['sisdri_target'][i]
            for j, interval in enumerate(duration_x):
                if d > interval[0] and d <= interval[1]:
                    duration_y[j].append(sisdri)
        x = []
        y = []
        err = []
        for i in range(len(duration_x)):
            x.append(np.mean(duration_x[i]))
            y.append(np.mean(duration_y[i]))
            err.append(np.std(duration_y[i]))
        plt.errorbar(x, y, yerr=err, fmt='o-', capsize=5)
        # plt.scatter(results_df['self_duration'], results_df['sisdri_target'],)
        plt.xlabel('Duration of reference speech(s)', fontsize = 14)
        plt.ylabel('SI-SDRi (dB)', fontsize = 14)
        plt.ylim([-5, 14])
        plt.tight_layout()
        plt.savefig( os.path.join('./debug/self_duration.png') )
        plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=str, help='Directory with stored CSV file')
    args = parser.parse_args()

    main(args)