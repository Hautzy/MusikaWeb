coord_depth = 64  # Dimension of latent coordinate and style random vectors
lat_len = 256  # not small 256
coord_len = (lat_len // 2) * 3
lat_depth = 64  # Depth of generated latent vectors
hop = 256  # Hop size (window size = 4*hop)
sigma_rescale = 75.0  # Spectrogram sigma used to normalize
mu_rescale = 25.0  # Spectrogram mu used to normalize
shape = 128  # Length of spectrograms time axis
sr = 44100
mel_bins = 256