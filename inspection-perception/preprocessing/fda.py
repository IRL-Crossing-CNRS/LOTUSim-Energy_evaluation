import numpy as np
from PIL import Image

def fda(src_img, trg_img, beta=0.01):
    """
    src_img, trg_img: HxWx3 numpy arrays, same size, values 0-255 (uint8 or float)
    beta: fraction of spectrum (centered low-freq square) to swap. 
          Start small (0.01-0.05) and increase if the effect is too weak.
    Returns: src_img with low-freq amplitude swapped toward trg_img's style.
    """
    src = src_img.astype(np.float32).transpose(2, 0, 1)  # C,H,W
    trg = trg_img.astype(np.float32).transpose(2, 0, 1)

    fft_src = np.fft.fft2(src, axes=(-2, -1))
    fft_trg = np.fft.fft2(trg, axes=(-2, -1))

    amp_src, pha_src = np.abs(fft_src), np.angle(fft_src)
    amp_trg = np.abs(fft_trg)

    amp_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    amp_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, h, w = amp_src.shape
    b = int(np.floor(min(h, w) * beta))
    cy, cx = h // 2, w // 2

    amp_src[:, cy-b:cy+b, cx-b:cx+b] = amp_trg[:, cy-b:cy+b, cx-b:cx+b]

    amp_src = np.fft.ifftshift(amp_src, axes=(-2, -1))
    fft_src_new = amp_src * np.exp(1j * pha_src)
    src_new = np.fft.ifft2(fft_src_new, axes=(-2, -1)).real

    src_new = np.clip(src_new.transpose(1, 2, 0), 0, 255).astype(np.uint8)
    return src_new


if __name__ == "__main__":
    sim = np.array(Image.open("test.jpg").convert("RGB"))
    real = np.array(Image.open("real_image.png").convert("RGB").resize(sim.shape[1::-1]))

    out = fda(sim, real, beta=0.03)
    Image.fromarray(out).save("sim_crack_fda.jpg")