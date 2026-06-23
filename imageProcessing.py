import cv2
import time
import os
import runtime as torcpy

# Force single-threaded execution for underlying libraries to prevent CPU oversubscription
cv2.setNumThreads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def denoise(image):
    # High-Granularity Compute: Non-Local Means Denoising
    denoised = cv2.fastNlMeansDenoisingColored(
        image, None, h=10, hColor=10,
        templateWindowSize=7, searchWindowSize=20
    )

    # Medium Compute: Sharpening Filter
    blurred = cv2.GaussianBlur(denoised, (9, 9), 10.0)
    sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)

    return sharpened


def extract_video_frames(video_path, max_frames=24):
    print(f"[Rank 0] Opening video file: {video_path}...", flush=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file at {video_path}.")

    frames = []
    frame_count = 0

    print(f"[Rank 0] Extracting up to {max_frames} frames", flush=True)
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1

    cap.release()
    print(f"[Rank 0] Successfully extracted {len(frames)} frames.", flush=True)
    return frames


def main():
    video_filename = "input_noisy.mp4"
    output_directory = "./output_frames"
    max_frames_to_process = 40

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    try:
        images = extract_video_frames(video_filename, max_frames=max_frames_to_process)
    except Exception as e:
        print(f"[Rank 0] ERROR loading video: {e}", flush=True)
        return

    print("\n" + "=" * 50)
    print(f"STARTING BENCHMARK RUN USING SCHEDULER: {torcpy.TORC_SCHEDULING.upper()}")
    print("=" * 50)

    start_time = time.time()
    tasks = []

    for i, img in enumerate(images):
        t0 = time.time()
        task = torcpy.submit(denoise, img)

        print(f"submit {i}: {(time.time() - t0):.3f}s")
        tasks.append(task)
        # print(f"  -> Submitted frame {i + 1}/{len(images)} to the scheduler", flush=True)

    # print("[Rank 0] All tasks submitted. Awaiting cluster completion...", flush=True)
    torcpy.waitall()

    end_time = time.time()
    total_duration = end_time - start_time

    print("[Rank 0] Gathering results and saving denoised frames to disk...", flush=True)
    for frame_idx, task in enumerate(tasks):
        processed_frame = task.result()
        output_path = os.path.join(output_directory, f"denoised_frame_{frame_idx:04d}.jpg")
        cv2.imwrite(output_path, processed_frame)

    print("\n" + "=" * 50, flush=True)
    print(f"BENCHMARK COMPLETED SUCCESSFULLY", flush=True)
    print(f"Total processing time: {total_duration:.2f} seconds", flush=True)
    print(f"Average throughput: {total_duration / len(images):.3f} seconds per frame", flush=True)
    print(f"All denoised frames saved by Rank 0 to: {output_directory}", flush=True)
    print("=" * 50 + "\n", flush=True)


if __name__ == "__main__":
    torcpy.start(main, profile="cpu")
