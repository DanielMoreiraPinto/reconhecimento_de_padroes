import setup_path

from denoiser.denoiser import training, test_denoising

if __name__ == "__main__":
    training()
    # training(ckpt_path=MODEL_PATH)
    # test_denoising(TEST_SAVE_PATH, TEST_SAMPLES_PATH)
    pass