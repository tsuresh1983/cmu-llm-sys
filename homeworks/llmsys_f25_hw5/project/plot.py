import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np

def plot(means, stds, labels, fig_name, title):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel(title)
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

# Fill the data points here
if __name__ == '__main__':
    single_mean, single_std = 14.98754370212555, 0.14124251494915885
    device0_mean, device0_std = 7.718271422386169, 0.23906278408017673
    device1_mean, device1_std = 8.201568818092346, 0.2932035662677353
    plot([device0_mean, device1_mean, single_mean],
        [device0_std, device1_std, single_std],
        ['Data Parallel - GPU0', 'Data Parallel - GPU1', 'Single GPU'],
        'ddp_vs_rn.png', title="GPT2 Execution Time (Second) [H100]")
    
    single_tps_mean, single_tps_std = 254025.62648902385, 938.4933482871502
    device0_tps_mean, device0_tps_std = 250288.07945994864, 1813.888516128469
    device1_tps_mean, device1_tps_std = 250915.35066842864, 1961.6435305511584
    plot([(device0_tps_mean + device1_tps_mean) , single_tps_mean],
        [(device0_tps_std**2 + device1_tps_std**2)**0.5, single_tps_std],
        ['Data Parallel - 2GPUs', 'Single GPU'],
        'ddp_vs_rn_tps.png', title="GPT2 Tokens per Second [H100]")

    pp_mean, pp_std = 41.888131499290466, 0.997773289680481
    mp_mean, mp_std = 36.94660031795502, 0.2459026575088501
    plot([pp_mean, mp_mean],
        [pp_std, mp_std],
        ['Pipeline Parallel', 'Model Parallel'],
        'pp_vs_mp.png', title="GPT2 Execution Time(seconds) [A100]")
    
    pp_mean, pp_std = 15287.464799959647, 364.1466806555618
    mp_mean, mp_std = 17323.064802567333, 115.29579540446139
    plot([pp_mean, mp_mean],
        [pp_std, mp_std],
        ['Pipeline Parallel', 'Model Parallel'],
        'pp_vs_mp_tps.png', title="GPT2 tokens per seconds [A100]")