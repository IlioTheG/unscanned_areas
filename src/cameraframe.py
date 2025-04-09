import numpy as np
import cv2 as cv
import datetime


class Image:
    """
    Represents an image loaded from bytes.

    Attributes:
        bytes (bytes): The raw image data as bytes.
        img (numpy.ndarray): The image as a NumPy array (BGR format).

    Methods:
        image_deserializer(): Deserialize the image bytes into a NumPy array.
        display(): Display the image using OpenCV.

    Example:
        image_bytes = b"..."  # Replace with actual image bytes
        image_instance = Image(image_bytes)
        image_instance.display()
    """

    def __init__(self, image_bytes:bytes, is_depth:bool) -> None:
        self.bytes = image_bytes
        self.is_depth = is_depth
        self.img = self.image_deserialiser()
        self.height = self.img.shape[0]
        self.width = self.img.shape[1]

    def __str__(self) -> str:
        return f"height: {self.height} x width: {self.width}"

    def image_deserialiser(self):
        """Image deserializer"""
        if self.is_depth == 1:
            nparr = np.frombuffer(self.bytes, np.uint8)
            img = cv.imdecode(nparr, cv.IMREAD_UNCHANGED)

            if img.dtype == np.uint8 and img.shape[2] == 4:
                depth = np.zeros(img.shape[:2], dtype=np.float32)
                depth.data = img.data
                d = depth.copy()
                distance_threshold = 5.0
                d[d > distance_threshold] = np.nan

                # Manually scale the depth values to the 0-255 range
                d_scaled = (d / distance_threshold) * 255.0

                # Convert to 8-bit image
                d_8bit = np.uint8(d_scaled)

                # resize depth to 720x960 depending on the orientation
                if d_8bit.shape[0] == 192 and d_8bit.shape[1] == 256:
                    d_8bit = cv.resize(d_8bit, (960, 720), interpolation=cv.INTER_NEAREST)
                else:
                    d_8bit = cv.resize(d_8bit, (720, 960), interpolation=cv.INTER_NEAREST)
                return d_8bit
            else:
                print('Depth image is not 4-channel')
                return img
        else:
            nparr = np.frombuffer(self.bytes, np.uint8)
            img = cv.imdecode(nparr, cv.IMREAD_UNCHANGED)
            # chek if image is not grayscale
            if len(img.shape) == 2:
                img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)  # Check if it is necessary
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                return img
            else:
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                return img

    def resize(self, height, width):
        self.img = cv.resize(self.img, (height, width), interpolation=cv.INTER_LINEAR)
        self.height = self.img.shape[0]
        self.width = self.img.shape[1]

    def grayscale(self):
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

    def display(self):
        cv.imshow('image', self.img)
        cv.waitKey(2000)
        cv.imwrite(f'./figures/image{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png', self.img)


class Camera:
    """
    Represents a camera with intrinsic parameters.

    Attributes:
        id (int): The camera's unique identifier.
        is_depth (int): Flag indicating whether the camera captures depth information (0 or 1).
        fx (float): Focal length in the x-direction.
        fy (float): Focal length in the y-direction.
        cx (float): Principal point (x-coordinate).
        cy (float): Principal point (y-coordinate).
    """
    def __init__(self, camera_id:int, is_depth:int, fx:float, fy:float, cx:float, cy:float) -> None:
        self.camera_id = camera_id
        self.is_depth = is_depth
        self.fx:float = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def __str__(self) -> str:
        return f"camera_id: {self.camera_id} | is_depth: {self.is_depth}\nfx: {self.fx}\nfy: {self.fy}\ncx: {self.cx}\ncy: {self.cy}"


class CameraFrame(Camera, Image): # camera_frame
    """
    Represents a captured image and a camera.

    Attributes:
        Inherits attributes from Camera and Image classes.
        image (Image): The captured image.

    Methods:
        Inherits methods from Camera and Image classes.

    Example:
        capture_0 = Capture(id=0, is_depth=0, fx=1.0, fy=1.0, cx=0.0, cy=0.0, image=image_bytes)
        capture_0.display()
    """
    def __init__(self, camera_id:int, is_depth:int, fx:float, fy:float, cx:float, cy:float, image:bytes) -> None:
        Camera.__init__(self, camera_id, is_depth, fx, fy, cx, cy)
        Image.__init__(self, image, is_depth)

    def __str__(self) -> str:
        return super(Camera, self).__str__() + ' and ' + super(Image, self).__str__()
