from cam import CameraStream

def main():
    CAMERA_SOURCE = "http://192.168.31.102:4747/video"
    stream = CameraStream(source=CAMERA_SOURCE, skip_frames=3)
    stream.run()


if __name__ == "__main__":
    main()