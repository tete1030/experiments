%run -d start.py resume stretch -r checkpoint_14.pth.tar
break experiments/stretch.py:153

import lib.utils.imutils
reload(lib.utils.imutils)
from lib.utils.imutils import show_heatmap, batch_resize

fig, axes = show_heatmap(img, n_rows=3, n_cols=4, transpose=(0,2,3,1), mean=self.train_dataset.mean, show=False)
fig, axes = show_heatmap(batch_resize(mask, (256, 256)).reshape(-1, 256, 256), fig=fig, axes=axes, v_min=0, v_max=1, alpha=0.5, show=True)
fig, axes = show_heatmap(batch_resize(locate_map_gt, (256, 256)).reshape(-1, 256, 256), fig=fig, axes=axes, v_min=0, v_max=1, alpha=0.5, show=True)
fig, axes = show_heatmap(posemap[:, 0], fig=fig, axes=axes, v_min=0, v_max=1, alpha=0.5, show=True)
fig, axes = show_heatmap(posemap[:, 17], fig=fig, axes=axes, alpha=0.5, show=True)

# used when DataParallel removed
checkpoint['state_dict'] = {k.replace(".module", ""):v for k, v in checkpoint['state_dict'].iteritems()}

myparams = list(exp.model[1].parameters())
[(~mp.grad.eq(0).data).sum() > 0 if mp.grad is not None else None for mp in myparams]
