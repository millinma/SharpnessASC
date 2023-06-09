import audmetric
import audobject
import torch
import tqdm


def transfer_features(features, device):
    return features.to(device).float()


def evaluate_categorical(model, device, loader, transfer_func, disable, criterion):
    metrics = {
        'UAR': audmetric.unweighted_average_recall,
        'ACC': audmetric.accuracy,
        'F1': audmetric.unweighted_average_fscore
    }

    model.to(device)
    model.eval()

    outputs = torch.zeros((len(loader.dataset), model.output_dim))
    targets = torch.zeros(len(loader.dataset))
    with torch.no_grad():
        for index, (features, target) in tqdm.tqdm(
            enumerate(loader),
            desc='Batch',
            total=len(loader),
            disable=disable,
        ):
            start_index = index * loader.batch_size
            end_index = (index + 1) * loader.batch_size
            if end_index > len(loader.dataset):
                end_index = len(loader.dataset)
            outputs[start_index:end_index, :] = model(
                transfer_func(features, device))
            targets[start_index:end_index] = target
    loss = criterion(outputs, targets.type(torch.LongTensor))
    targets = targets.numpy()
    outputs = outputs.cpu()
    predictions = outputs.argmax(dim=1).numpy()
    outputs = outputs.numpy()
    return {
        key: metrics[key](targets, predictions)
        for key in metrics.keys()
    }, targets, predictions, outputs, loss.item()


class LabelEncoder(audobject.Object):
    def __init__(self, labels):
        self.labels = sorted(labels)
        codes = range(len(labels))
        self.inverse_map = {code: label for code,
                            label in zip(codes, labels)}
        self.map = {label: code for code,
                    label in zip(codes, labels)}

    def encode(self, x):
        return self.map[x]

    def decode(self, x):
        return self.inverse_map[x]


def disaggregated_evaluation(df, groundtruth, task, stratify, evaluation_type: str = 'regression'):
    if evaluation_type == 'regression':
        metrics = {
            'CC': audmetric.pearson_cc,
            'CCC': audmetric.concordance_cc,
            'MSE': audmetric.mean_squared_error,
            'MAE': audmetric.mean_absolute_error
        }
    elif evaluation_type == 'categorical':
        metrics = {
            'UAR': audmetric.unweighted_average_recall,
            'ACC': audmetric.accuracy,
            'F1': audmetric.unweighted_average_fscore
        }
    else:
        raise NotImplementedError(evaluation_type)

    df = df.reindex(groundtruth.index)
    results = {key: {} for key in metrics.keys()}
    for key in metrics.keys():
        results[key]['all'] = metrics[key](
            groundtruth[task],
            df['predictions']
        )
        for stratifier in stratify:
            for variable in groundtruth[stratifier].unique():
                indices = groundtruth.loc[groundtruth[stratifier]
                                          == variable].index
                results[key][variable] = metrics[key](
                    groundtruth.reindex(indices)[task],
                    df.reindex(indices)['predictions']
                )

    return results
