import matplotlib.pyplot as plt

def viz(i=4, h=[8, 4], o=2, acts=['ReLU', 'ReLU']):
    layers = [i] + h + [o]
    labels = ['Input'] + [f'H{i+1}' for i in range(len(h))] + ['Out']
    act_labels = acts + ['None']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Neural Network Architecture')
    for li, n in enumerate(layers):
        for ni in range(n):
            ax.add_patch(plt.Circle((li * 2, ni - n/2), 0.3, fill=False))
    for li in range(len(layers) - 1):
        for f in range(layers[li]):
            for t in range(layers[li + 1]):
                ax.plot([li * 2 + 0.3, (li + 1) * 2 - 0.3],
                        [f - layers[li]/2, t - layers[li + 1]/2],
                        'k-', alpha=0.1)
    for li, lbl in enumerate(labels):
        ax.text(li * 2, layers[li]/2 + 0.5, lbl + f'\n({layers[li]})', ha='center')
        if li < len(act_labels):
            ax.text(li * 2 + 1, layers[li]/2 + 0.5, act_labels[li], ha='center', color='blue')
    ax.axis('off')
    plt.show()

viz()