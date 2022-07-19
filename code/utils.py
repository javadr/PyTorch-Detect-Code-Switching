import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score

from data import CFG

def res_plot(data, desc='', p=3):
    legend=['Train','Test']
    fig, axes = plt.subplots(1,2, figsize = (17,6), facecolor=(1,1,1))
    for i,title in enumerate(['Loss', 'F1 Score']):
        axes[i].set_ylabel(title)
    epochs = len(data['train_loss'])

    for idx, t in enumerate(['loss', 'f1']):
        x = range(1,epochs+1)
        ax = axes[idx]
        ax.set_xticks(x[::p])
        ax.set_xticklabels(ax.get_xticks(), rotation = 90)
        # title = 'Loss' if idx==0 else 'F1 Score'
        # ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.plot(x, data[f'train_{t}'], '-', label=legend[0])
        ax.plot(x, data[f'val_{t}'], '-', label=legend[1])
        c = 0
        for i,j in list(zip( [id for tup in [(i,i) for i in x[::p][:-1]] for id in tup],
                    [item for tup in zip(data[f'train_{t}'][::p][:-1],data[f'val_{t}'][::p][:-1]) for item in tup] )):
            c = 1-c
            ax.annotate(f"{j:.3f}", xy=(i,j), rotation=60, va=['top','bottom'][c] if idx else ['bottom','top'][c], color=['g', 'k'][c])
        ax.annotate(f"{data[f'train_{t}'][-1]:.{max(4,p)}f}", xy=(epochs,data[f'train_{t}'][-1]), color='r')
        ax.annotate(f"{data[f'val_{t}'][-1]:.{max(4,p)}f}", xy=(epochs,data[f'val_{t}'][-1]), color='r')
        notch = (max if idx else min)(data[f'val_{t}'])
        ax.plot([notch, len(data[f'val_{t}'])], [notch, notch], 'gray')
        ax.annotate(f"{notch:.4f}", xy=(1,notch), ha='right', c='black')
        ax.legend()
    metric_name = f'../images/plot[{datetime.now().strftime("%y%m%d%H%M")}]-Ep{epochs}B{CFG.batch_size}{desc}.png'
    fig.suptitle(desc, fontsize=16)
    fig.savefig(metric_name, bbox_inches='tight', dpi=100)
    plt.show()


flatten = lambda tensor: tensor.view(-1).detach().cpu().numpy()

def evaluation(y_true, y_pred, metrics):
    output = {}
    if "accuracy" in metrics:
        output["accuracy"] = accuracy_score(y_true, y_pred)
    if "f1" in metrics:
        output["f1"] = f1_score(y_true, y_pred, average="macro")
    return output

