import utils
from collections import defaultdict
from datetime import datetime
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch


def validation(args, model, test_pts, logger, batch_n, epoch):
    model.eval()
    all_raw_preds, all_preds, all_labels = [], [], []
    for as_, bs_, labels in test_pts:
        if as_:
            as_ = as_.to(utils.get_device())
            bs_ = bs_.to(utils.get_device())
            labels = labels.to(utils.get_device())

        with torch.no_grad():
            if as_:
                emb_as_ = model.emb_model(
                    as_.x, as_.edge_index, as_.edge_attr, as_.batch
                )
                emb_bs_ = model.emb_model(
                    bs_.x, bs_.edge_index, bs_.edge_attr, bs_.batch
                )

            pred = model(emb_as_, emb_bs_)
            raw_pred = model.predict(pred)

            pred = model.clf_model(raw_pred.unsqueeze(1)).argmax(dim=-1)
            pred = pred.cpu()
            raw_pred = raw_pred.cpu()
            raw_pred *= -1

        all_raw_preds.append(raw_pred)
        all_preds.append(pred)
        all_labels.append(labels)
    pred = torch.cat(all_preds, dim=-1)
    labels = torch.cat(all_labels, dim=-1).cpu()
    raw_pred = torch.cat(all_raw_preds, dim=-1)
    acc = torch.mean((pred == labels).type(torch.float))
    prec = (
        torch.sum(pred * labels).item() / torch.sum(pred).item()
        if torch.sum(pred) > 0
        else float("NaN")
    )
    recall = (
        torch.sum(pred * labels).item() / torch.sum(labels).item()
        if torch.sum(labels) > 0
        else float("NaN")
    )
    labels = labels.detach().cpu().numpy()
    raw_pred = raw_pred.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    auroc = roc_auc_score(labels, raw_pred)
    avg_prec = average_precision_score(labels, raw_pred)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    
    # if verbose:
    #     import matplotlib.pyplot as plt

    #     precs, recalls, threshs = precision_recall_curve(labels, raw_pred)
    #     plt.plot(recalls, precs)
    #     plt.xlabel("Recall")
    #     plt.ylabel("Precision")
    #     plot_path = f"plots/precision-recall-curve-tag{args.tag}.png"
    #     plt.savefig(plot_path)
    #     print(f"Saved PR curve plot in {plot_path}")

    print("\n{}".format(str(datetime.now())))
    print(
        "Validation. Epoch {}. Acc: {:.4f}. "
        "P: {:.4f}. R: {:.4f}. AUROC: {:.4f}. AP: {:.4f}.\n     "
        "TN: {}. FP: {}. FN: {}. TP: {}".format(
            epoch, acc, prec, recall, auroc, avg_prec, tn, fp, fn, tp
        )
    )

    if not args.test:
        logger.add_scalar("Accuracy/test", acc, batch_n)
        logger.add_scalar("Precision/test", prec, batch_n)
        logger.add_scalar("Recall/test", recall, batch_n)
        logger.add_scalar("AUROC/test", auroc, batch_n)
        logger.add_scalar("AvgPrec/test", avg_prec, batch_n)
        logger.add_scalar("TP/test", tp, batch_n)
        logger.add_scalar("TN/test", tn, batch_n)
        logger.add_scalar("FP/test", fp, batch_n)
        logger.add_scalar("FN/test", fn, batch_n)

        print(f"Saving ckpt to {args.model_path}")
        torch.save(model.state_dict(), args.model_path)
