import cv2
import numpy as np
import time
import runtime as torcpy


def denoise(img_array):
    # application logic. CPU intensive algorithm to denoise set of pictures

    denoised = cv2.fastNlMeansDenoisingColored(img_array, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=15)
    blurred = cv2.GaussianBlur(denoised, (9, 9), 10.0)
    sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)

    return True


def generate_synthetic_dataset(num_images=16):
    """
    Generates a homogeneous batch of 4K images directly in memory on Rank 0.
    """
    # flush=True forces the output through the MPI buffer immediately
    print(f"[Rank 0] Generating {num_images} synthetic 4K images in RAM...", flush=True)
    dataset = []
    # Generate the heavy random math EXACTLY ONCE
    base_img = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
    for _ in range(num_images):
        # Copying the memory block is near-instant
        dataset.append(base_img.copy())
    print(f"[Rank 0] Dataset generated successfully!", flush=True)
    return dataset


def main():
    # number of images to process
    num_images = 20
    images = generate_synthetic_dataset(num_images)

    print("\n" + "=" * 50)
    print(f"STARTING BENCHMARK RUN USING SCHEDULER: {torcpy.TORC_SCHEDULING.upper()}")
    print("=" * 50)

    start_time = time.time()
    tasks = []

    for i, img in enumerate(images):
        task = torcpy.submit(denoise, img)
        tasks.append(task)
        print(f"  -> Submitted task {i + 1}/{num_images} to the scheduler", flush=True)

    print("[Rank 0] All tasks submitted. Awaiting cluster completion...", flush=True)
    torcpy.waitall()

    end_time = time.time()
    total_duration = end_time - start_time

    print("\n" + "=" * 50)
    print(f"BENCHMARK COMPLETED SUCCESSFULLY")
    print(f"Total processing time: {total_duration:.2f} seconds")
    print(f"Average throughput: {total_duration / num_images:.3f} seconds per image")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    torcpy.start(main, profile="cpu")
