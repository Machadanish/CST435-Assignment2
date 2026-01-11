import cv2
import numpy as np
import os
import time
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# --- SECTION 1: CORE IMAGE PROCESSING PIPELINE ---
def apply_filters(image_path):
    """
    Applies 5 filters to a single image.
    1. Grayscale
    2. Gaussian Blur (3x3)
    3. Edge Detection (Sobel)
    4. Sharpening
    5. Brightness Adjustment
    """
    output_dir = "processed_images"
  
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except FileExistsError:
            pass
    
    try:
        img = cv2.imread(image_path)
        if img is None: 
            return None

        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Gaussian Blur (3x3)
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        
        # 3. Edge Detection (Sobel)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobelx, sobely)
        
        # 4. Sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(img, -1, kernel)
        
        # 5. Brightness (+50)
        brightness = cv2.convertScaleAbs(img, alpha=1, beta=50)
        
    
        filename = os.path.basename(image_path)
        cv2.imwrite(os.path.join(output_dir, f"out_{filename}"), sharpened)
        return filename
    except Exception as e:
        return None

# --- SECTION 2: PARALLEL PARADIGM A (Multiprocessing) ---
def run_multiprocessing(image_list, num_workers):
    with mp.Pool(processes=num_workers) as pool:
        pool.map(apply_filters, image_list)

# --- SECTION 3: PARALLEL PARADIGM B (Concurrent Futures) ---
def run_concurrent_futures(image_list, num_workers):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Convert to list to force execution of the generator
        list(executor.map(apply_filters, image_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Image Processing Script")
    parser.add_argument('--mode', choices=['mp', 'cf'], required=True, 
                        help="mp: Multiprocessing, cf: Concurrent Futures")
    parser.add_argument('--workers', type=int, default=4, 
                        help="Number of parallel processes")
    args = parser.parse_args()


    input_folder = "images_subset"
    output_folder = "processed_images"

    if not os.path.exists(input_folder):
        print(f"Error: Folder '{input_folder}' not found.")
        exit()


    os.makedirs(output_folder, exist_ok=True)
        

    print(f"Scanning {input_folder} for images...")
    images = []
    for root, dirs, files in os.walk(input_folder):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(root, f))


    num_images = len(images)
    
    if num_images == 0:
        print("No images found in the directory.")
        exit()

    print(f"Processing {num_images} images with {args.workers} workers using '{args.mode}' mode...")
    
    start_time = time.time()
    
    if args.mode == 'mp':
        run_multiprocessing(images, args.workers)
    elif args.mode == 'cf':
        run_concurrent_futures(images, args.workers)
        
    duration = time.time() - start_time
    print("-" * 30)
    print(f"RESULTS FOR {args.mode.upper()} MODE")
    print(f"Workers: {args.workers}")
    print(f"Time Taken: {duration:.4f} seconds")

    print("-" * 30)
