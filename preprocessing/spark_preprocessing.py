from pyspark.sql import SparkSession
import cv2

# Example: parallelize simple frame count

def process_video(path: str):
    cap = cv2.VideoCapture(path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return path, frames

if __name__ == '__main__':
    spark = SparkSession.builder.appName('VideoPreprocessing').getOrCreate()
    sc = spark.sparkContext

    # Assume a text file with one video path per line
    video_paths = sc.textFile('video_paths.txt')
    results = video_paths.map(process_video).collect()

    for p, c in results:
        print(f"{p}: {c} frames")