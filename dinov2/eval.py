

def dino_evaluate(
    model, logreg, gt_img, eval_dir, scribbles_2d, rgb_img, rgb_img_scribbles=None, sliding_window=None, k_best_iou=None,
):
    if sliding_window is not None:
        sliding_window = config_to_instance(**sliding_window)
    c, H, W = rgb_img.shape
    if sliding_window is not None:
        Hr, Wr = (H // 14) * 14, (W // 14) * 14
        img = F.interpolate(
            input=rgb_img[None], size=(Hr, Wr), mode="bilinear"
        ).squeeze()
        ftest = sliding_window(img, model.predict)
        c, h, w = ftest.shape
        pca, _ = pca_on_embeddings([ftest], 3, max_pred=100000)
        save_pcas(torch.from_numpy(pca[0]).cuda(), eval_dir)
    else:
        ftest, _ = model.predict(rgb_img)
    if logreg is None:
        if sliding_window is not None:
            Hr, Wr = (H // 14) * 14, (W // 14) * 14
            img_scribbles = F.interpolate(
                input=rgb_img_scribbles[None], size=(Hr, Wr), mode="bilinear"
            ).squeeze()
            fscribbles = sliding_window(img_scribbles, model.predict)
        else:
            fscribbles, (H, W) = model.predict(rgb_img_scribbles)
        scribbles_2d = resize(scribbles_2d[None], tuple(fscribbles.shape[1:]))
        print(fscribbles.shape, scribbles_2d.shape)
        fscribbles = fscribbles.view(len(fscribbles), -1).T
        fscribbles /= fscribbles.norm(dim=1, keepdim=True) + 1e-8
        logreg = LogisticRegression(C=1, max_iter=1000).fit(
            fscribbles.cpu().numpy(),
            scribbles_2d.flatten().cpu().numpy().astype(int),
        )
    print(
        "Evaluating segmentation from bilinear interpolation of DINOv2 embeddings."
    )
    c, h, w = ftest.shape
    pca, _ = pca_on_embeddings([ftest], 3)
    save_pcas(torch.from_numpy(pca[0]).cuda(), eval_dir)
    # save_img(os.path.join(eval_dir, "pca.png"), None, viz_normalization(torch.from_numpy(pca[0]).cuda(), dim=(1,2)))
    ftest /= ftest.norm(dim=0, keepdim=True) + 1e-8
    anchor = logreg.predict_proba(ftest.view(len(ftest), -1).T.cpu().numpy())[:, 1]
    anchor = torch.from_numpy(anchor).view(h, w).cuda()
    anchor = resize(anchor[None], (H, W)).squeeze()
    print("Using k_best_iou", k_best_iou)
    best_iou, k_best_iou = segmentation_2d(
        anchor, gt_img, rgb_img, eval_dir, k_best_iou
    )
    return best_iou, k_best_iou, logreg
