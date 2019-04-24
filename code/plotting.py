import plotly.graph_objs as go
from plotly.offline import iplot
import matplotlib.pyplot as plt

def plot_state_prob(prob, title):

    data = []
    updatemenus = dict(active=1, buttons=[])

    for i in range(prob.shape[0]):
        data.append(go.Heatmap(z=prob[i, ...].T, colorscale='Viridis'))
        updatemenus['buttons'].append(dict(label='Trial {0}'.format(i),
                                           method='update',
                                           args=[{'visible': [True if j == i else False for j in
                                                              range(prob.shape[0])]},
                                                 {
                                                     'title': 'Trial {0} state activation probabilities - '.format(
                                                         i, title)}]))

    updatemenus = [updatemenus]
    layout = dict(
        title='Trial {0} state activation probabilities - {0}'.format(prob.shape[0], title),
        showlegend=False,
        updatemenus=updatemenus, xaxis=dict(title='Time point'), yaxis=dict(title='State'))

    fig = dict(data=data, layout=layout)
    iplot(fig, filename='update_dropdown')


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, figsize=(7, 7)):

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()