sims_outfile = open("C:/Users/jseong/Desktop/hwnam/2026/lights2/step-2-CAE-GMM/0_CAE_Clustering_100times/global_results_0422"
"/output100.txt", 'w')
#%%
# CNN Autoencoder with clustering for VIIRS hotspot maps
# Using balanced encoder-decoder: filters (16, 32, 64), latent_dim=64
# Explicitly treating pixel value -1 as nodata

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

import glob
import re
import random
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#%%
for sim_run in range (1, 101):
    # -----------------------------
    # Reproducibility
    # -----------------------------
    os.environ["PYTHONHASHSEED"] = str(sim_run)
    my_rand_num = sim_run
    np.random.seed(my_rand_num)
    random.seed(my_rand_num)
    tf.random.set_seed(my_rand_num)
    
    # -----------------------------
    # User paths and parameters
    # -----------------------------
    input_dir = r"C:/Users/jseong/Desktop/hwnam/2026/lights2/step-2-CAE-GMM/0_CAE_Clustering_100times/01_resampled-10km-negative-one-as-nodata"
    output_dir = r"C:/Users/jseong/Desktop/hwnam/2026/lights2/step-2-CAE-GMM/0_CAE_Clustering_100times/global_results_0422/"+ str(sim_run)
    
    os.makedirs(output_dir, exist_ok=True)
    output_plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(output_plot_dir, exist_ok=True)
    
    csv_outfile = os.path.join(output_dir, "image_cluster_memberships.csv")
    
    latent_dim = 64
    batch_size = 8
    epochs = 300
    patience = 20
    
    k_min, k_max = 2, 10
    random_state = my_rand_num
    
    # -----------------------------
    # Helpers
    # -----------------------------
    def pad_to_multiple_of_8(img, mask):
        H, W = img.shape
        new_H = ((H + 7) // 8) * 8
        new_W = ((W + 7) // 8) * 8
        pad_h = new_H - H
        pad_w = new_W - W
        img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        mask_padded = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        return img_padded, mask_padded
    
    def read_image_and_mask(path):
        with rasterio.open(path) as src:
            arr = src.read(1).astype("float32")
            # Explicit nodata handling: -1 always means nodata
            mask = arr != -1
            filled = np.where(mask, arr, 0.0)
            filled, mask = pad_to_multiple_of_8(filled, mask.astype("float32"))
            return filled, mask, src.profile
    
    # -----------------------------
    # Load images
    # -----------------------------
    pattern = re.compile(r"(\d{4})(\d{2})\.tif$")
    files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    
    image_names, years, months, images, masks, profiles = [], [], [], [], [], []
    
    for fpath in tqdm(files, desc="Loading images"):
        fname = os.path.basename(fpath)
        m = pattern.match(fname)
        if not m:
            continue
        yy, mm = int(m.group(1)), int(m.group(2))
        img, mask, profile = read_image_and_mask(fpath)
        images.append(img)
        masks.append(mask)
        profiles.append(profile)
        image_names.append(fname)
        years.append(yy)
        months.append(mm)
    
    # -----------------------------
    # Save padded images
    # -----------------------------
    padded_output_dir = os.path.join(output_dir, "padded_images_multiple_of_8")
    os.makedirs(padded_output_dir, exist_ok=True)
    
    for img, profile, fname in zip(images, profiles, image_names):
        padded_path = os.path.join(padded_output_dir, fname)
        profile_updated = profile.copy()
        profile_updated.update({
            "height": img.shape[0],
            "width": img.shape[1],
            "transform": rasterio.transform.from_origin(
                profile["transform"].c,
                profile["transform"].f,
                profile["transform"].a,
                -profile["transform"].e
            ),
            "nodata": -1  # ensure nodata is explicitly saved
        })
        with rasterio.open(padded_path, "w", **profile_updated) as dst:
            dst.write(img, 1)
    
    # -----------------------------
    # Prepare data
    # -----------------------------
    shapes = set([img.shape for img in images])
    if len(shapes) != 1:
        raise ValueError(f"Not all images padded to same shape. Shapes: {shapes}")
    H, W = images[0].shape
    
    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)
    
    # Global normalization using only valid pixels from all images
    images_norm = np.zeros_like(images, dtype="float32")

    all_valid = masks.astype(bool)
    global_vals = images[all_valid]

    if global_vals.size == 0:
        global_mean, global_std = 0.0, 1.0
    else:
        global_mean = float(global_vals.mean())
        global_std = float(global_vals.std())
        if global_std == 0:
            global_std = 1.0

    images_norm = (images - global_mean) / global_std
    images_norm[~all_valid] = 0.0
    
    # Input: normalized image + mask
    X = np.stack([images_norm, masks], axis=-1)
    
    # -----------------------------
    # Autoencoder
    # -----------------------------
    def build_autoencoder(input_shape, latent_dim):
        inp = Input(shape=input_shape)
    
        # Encoder
        x = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
    
        shape_before_flat = tf.keras.backend.int_shape(x)[1:]
        x = layers.Flatten()(x)
        x = layers.Dropout(0.1)(x)
        latent = layers.Dense(latent_dim, name="latent", kernel_regularizer=regularizers.l2(1e-6))(x)
    
        # Decoder
        x_dec = layers.Dense(np.prod(shape_before_flat), activation="relu", name="dec_dense")(latent)
        x_dec = layers.Reshape(shape_before_flat, name="dec_reshape")(x_dec)
        x_dec = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu", name="dec_deconv1")(x_dec)
        x_dec = layers.BatchNormalization(name="dec_bn1")(x_dec)
        x_dec = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu", name="dec_deconv2")(x_dec)
        x_dec = layers.BatchNormalization(name="dec_bn2")(x_dec)
        x_dec = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu", name="dec_deconv3")(x_dec)
        out = layers.Conv2D(1, 3, padding="same", activation="linear", name="reconstruction")(x_dec)
    
        autoencoder = Model(inp, out, name="autoencoder")
        encoder = Model(inp, latent, name="encoder")
    
        # Shared decoder
        latent_input = Input(shape=(latent_dim,), name="decoder_input")
        y = autoencoder.get_layer("dec_dense")(latent_input)
        y = autoencoder.get_layer("dec_reshape")(y)
        y = autoencoder.get_layer("dec_deconv1")(y)
        y = autoencoder.get_layer("dec_bn1")(y)
        y = autoencoder.get_layer("dec_deconv2")(y)
        y = autoencoder.get_layer("dec_bn2")(y)
        y = autoencoder.get_layer("dec_deconv3")(y)
        y = autoencoder.get_layer("reconstruction")(y)
        decoder = Model(latent_input, y, name="decoder")
    
        return autoencoder, encoder, decoder, shape_before_flat
    
    def masked_mse(y_true, y_pred):
        image_true = y_true[..., 0:1]
        mask = y_true[..., 1:2]
        diff = (image_true - y_pred) * mask
        sq = tf.square(diff)
        sum_sq = tf.reduce_sum(sq, axis=[1,2,3])
        count = tf.reduce_sum(mask, axis=[1,2,3])
        count = tf.where(count <= 0, tf.ones_like(count), count)
        return tf.reduce_mean(sum_sq / count)
    
    autoencoder, encoder, decoder, shape_before_flat = build_autoencoder((H, W, 2), latent_dim)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=masked_mse)
    
    # Dataset
    X_inp = X.astype("float32")
    Y_true = np.stack([images_norm, masks], axis=-1).astype("float32")
    dataset = tf.data.Dataset.from_tensor_slices((X_inp, Y_true)).batch(batch_size)
    
    # Training
    es = EarlyStopping(monitor="loss", patience=patience, restore_best_weights=True)
    ckpt_path = os.path.join(output_dir, "autoencoder_best_weights.weights.h5")
    mc = ModelCheckpoint(ckpt_path, monitor="loss", save_best_only=True, save_weights_only=True, verbose=0)
    
    history = autoencoder.fit(dataset, epochs=epochs, callbacks=[es, mc], verbose=1)
    
    # Save training loss
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Masked MSE Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.savefig(os.path.join(output_plot_dir, "training_loss.png"))
    plt.close()
    
    # Latent vectors
    latent_vectors = encoder.predict(X_inp, batch_size=batch_size)
    
    # -----------------------------
    # Clustering (GMM)
    # -----------------------------
    scaler = StandardScaler()
    Z = scaler.fit_transform(latent_vectors).astype(np.float64)
    k_range = range(k_min, k_max+1)
    
    bics, gmm_models = [], {}
    for k in k_range:
        gm = GaussianMixture(
            n_components=k,
            random_state=random_state,
            n_init=10,
            reg_covar=1e-5
        ).fit(Z)
        gmm_models[k] = gm
        bics.append(gm.bic(Z))
    
    plt.figure()
    plt.plot(list(k_range), bics, marker='o')
    plt.xlabel('k')
    plt.ylabel('BIC')
    plt.title('GMM BIC Scores')
    plt.savefig(os.path.join(output_plot_dir, "gmm_bic.png"))
    plt.close()
    
    best_k_gmm = list(k_range)[int(np.argmin(bics))]
    gmm_labels = gmm_models[best_k_gmm].predict(Z)
    
    # -----------------------------
    # GMM Stability Test (ARI/NMI)
    # -----------------------------
    n_repeats = 10
    all_labels = []
    for run in range(n_repeats):
        gm_tmp = GaussianMixture(
            n_components=best_k_gmm,
            random_state=run,
            n_init=10,
            reg_covar=1e-5
        ).fit(Z)
        all_labels.append(gm_tmp.predict(Z))
    
    aris, nmis = [], []
    for i in range(n_repeats):
        for j in range(i+1, n_repeats):
            aris.append(adjusted_rand_score(all_labels[i], all_labels[j]))
            nmis.append(normalized_mutual_info_score(all_labels[i], all_labels[j]))
    
    print(f"\n=== GMM Stability Test (k={best_k_gmm}) ===")
    print(f"ARI: mean={np.mean(aris):.3f}, std={np.std(aris):.3f}")
    print(f"NMI: mean={np.mean(nmis):.3f}, std={np.std(nmis):.3f}")
    
    sims_outfile.write(f"### my_rand_uum: {my_rand_num} ###")
    sims_outfile.write(f"ARI: mean={np.mean(aris):.3f}, std={np.std(aris):.3f}")
    sims_outfile.write(f"NMI: mean={np.mean(nmis):.3f}, std={np.std(nmis):.3f}")
    
    plt.figure()
    plt.boxplot([aris, nmis], labels=["ARI", "NMI"])
    plt.title(f"GMM Stability at k={best_k_gmm}")
    plt.savefig(os.path.join(output_plot_dir, "gmm_stability.png"))
    plt.close()
    
    # Robustness test against reg_covar
    reference_labels = GaussianMixture(
        n_components=best_k_gmm,
        random_state=999,
        n_init=10,
        reg_covar=1e-5
    ).fit(Z).predict(Z)
    
    for reg in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        aris, nmis = [], []
        for seed in range(50):
            gm = GaussianMixture(
                n_components=best_k_gmm,
                random_state=seed,
                n_init=10,
                reg_covar=reg
            ).fit(Z)
            labels = gm.predict(Z)
            aris.append(adjusted_rand_score(reference_labels, labels))
            nmis.append(normalized_mutual_info_score(reference_labels, labels))
        print(f"\n=== reg_covar={reg} ===")
        print(f"ARI: mean={np.mean(aris):.3f}, std={np.std(aris):.3f}")
        print(f"NMI: mean={np.mean(nmis):.3f}, std={np.std(nmis):.3f}")
        sims_outfile.write(f"\n=== reg_covar={reg} ===")
        sims_outfile.write(f"ARI: mean={np.mean(aris):.3f}, std={np.std(aris):.3f}")
        sims_outfile.write(f"NMI: mean={np.mean(nmis):.3f}, std={np.std(nmis):.3f}")
    
    # -----------------------------
    # Output table
    # -----------------------------
    df = pd.DataFrame({
        "imageName": image_names,
        "YYYY": years,
        "mm": months,
        "cluster_GMM": gmm_labels
    })
    df.to_csv(csv_outfile, index=False)
    
    # Pivot table
    gmm_table = df.pivot(index="YYYY", columns="mm", values="cluster_GMM")
    gmm_table = gmm_table.reindex(columns=range(1, 13))
    gmm_table.to_csv(os.path.join(output_dir, "gmm_monthly_table.csv"))
    print(gmm_table)
    sims_outfile.write(gmm_table.to_string(index=False))
    
    # -----------------------------
    # Decode cluster means
    # -----------------------------
    decoded_dir = os.path.join(output_dir, "decoded_images")
    os.makedirs(decoded_dir, exist_ok=True)
    
    autoencoder.load_weights(ckpt_path)
    
    cluster_means = []
    for k in range(best_k_gmm):
        idx = np.where(gmm_labels == k)[0]
        if idx.size == 0:
            print(f"Cluster {k} is empty — skipping")
            continue
        mean_latent = latent_vectors[idx].mean(axis=0)
        cluster_means.append((k, mean_latent))
    
    for k, mean_latent in cluster_means:
        decoded = decoder.predict(np.expand_dims(mean_latent, axis=0))[0, :, :, 0]
        profile = profiles[0].copy()
        profile.update({
            "height": H,
            "width": W,
            "count": 1,
            "dtype": "float32",
            "compress": "lzw",
            "nodata": -1
        })
        # restore nodata
        decoded_masked = np.where(masks[0] > 0, decoded, -1)
        out_path = os.path.join(decoded_dir, f"cluster_{k:02d}_decoded.tif")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(decoded_masked.astype("float32"), 1)
        print(f"Saved decoded cluster image: {out_path}")

#%%
sims_outfile.close()