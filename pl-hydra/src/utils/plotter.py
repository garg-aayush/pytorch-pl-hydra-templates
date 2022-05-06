import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_cm(cm, name_classes):
    cm_np = cm.cpu().numpy()
    df_cm = pd.DataFrame((cm_np/np.sum(cm_np))*10, 
            index = [i for i in name_classes],
            columns= [i for i in name_classes])
    
    plt.figure(figsize = (10,7))
    fig_ = sns.heatmap(df_cm, annot=True, cmap=None).get_figure()
    return fig_

def plot_preds(images, labels, preds, name_classes, nimg=32, ncols=8,
                data_mean=[], data_std=[]):
    nrows = nimg//ncols  
    # define figure
    fig_, axes=plt.subplots(nrows, ncols, figsize=(12, 8))
    axes = axes.ravel()

    #print(np.min(images), np.max(images))
    for i in range(nimg):
        label_name = name_classes[labels[i]]
        pred_name = name_classes[preds[i]]
        image = images[i]
        image[0] = image[0]*data_std[0] + data_mean[0]
        image[1] = image[1]*data_std[1] + data_mean[1]
        image[2] = image[1]*data_std[2] + data_mean[2]
        #print(np.min(image), np.max(image))
        image = np.transpose((image*255).astype('uint8'), (1,2,0))
        axes[i].imshow(image)
        axes[i].set_title(f'label: {label_name} \n pred: {pred_name}', fontsize=8)
        axes[i].axis('off')
    plt.subplots_adjust(hspace=0.2)
    return fig_ 