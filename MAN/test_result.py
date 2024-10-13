import numpy as np
from utils.tasks import lu_classify, predict_popus

latent_fusion = np.load("MAN/embed_result/best_embed.npy")

lu_scores = lu_classify(latent_fusion)
popus_scores = predict_popus(latent_fusion)
