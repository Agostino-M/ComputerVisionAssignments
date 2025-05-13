import cv2
import dlib
import numpy as np
import imageio
import argparse

class Rect():
    x: int
    y: int
    w: int
    h: int
    center: tuple

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.center = (x+w//2, y+h//2)

    def __repr__(self):
        return f"Rect(x={self.x}, y={self.y}, w={self.w}, h={self.h}, center={self.center})"

    def translate(self, t_x, t_y):
        self.x = round(self.x + t_x)
        self.y = round(self.y + t_y)
        self.center = (round(self.x + self.w // 2), round(self.y + self.h // 2))

    def scale(self, s_x, s_y):
        self.w = round(self.w * s_x)
        self.h = round(self.h * s_y)
        self.x = round(self.x * s_x)
        self.y = round(self.y * s_y)
        self.center = (round(self.x + self.w // 2), round(self.y + self.h // 2))

    def to_dlib_rectangle(self):
        return dlib.rectangle(
            left=self.x,
            top=self.y,
            right=self.x + self.w - 1,
            bottom=self.y + self.h - 1,
        )

def detect_face(classifier, image, window_name, print_debug) -> Rect:
    """ Function that detect faces in :image using a :classifier"""
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", frame_gray)
    rect_list = classifier.detectMultiScale(image_gray, minNeighbors=6, scaleFactor=1.1, minSize=(100,100))

    if len(rect_list) <= 0:
        raise RuntimeError(f"Cannot detect face in image: {window_name}")
    
    # the assumption is we must have only one face per image
    rect_list = sorted(rect_list, key=lambda r: r[2] * r[3], reverse=True)
    #x, y, w, h = rect_list[0]
    rect = Rect(*rect_list[0])

    if print_debug:
        print(f"Found {len(rect_list)} faces")
        im_copy = image.copy()
        cv2.rectangle(im_copy, (rect.x, rect.y), (rect.x+rect.w, rect.y+rect.h), color=(0, 0, 255), thickness=2)
        cv2.circle(im_copy, rect.center, radius=5, color=(0, 255, 0), thickness=-1)
        cv2.imshow(f"Detected face in {window_name}", im_copy)

    return rect

def align_faces(floating_im, reference_im, rect_fim: Rect, rect_rim: Rect, print_debug):
    """ Function that align floating image face with reference image face """

    assert floating_im.shape == reference_im.shape, "Images have different shapes"
    n_rows, n_cols = reference_im.shape[:2]

    # scaling: ratio between widths and heights
    s_x = rect_rim.w / rect_fim.w
    s_y = rect_rim.h / rect_fim.h
    rect_fim.scale(s_x, s_y)
    # traslation: diff betwseen centers
    t_x = rect_rim.center[0] - rect_fim.center[0]
    t_y = rect_rim.center[1] - rect_fim.center[1]
    rect_fim.translate(t_x, t_y)

    M = np.float32([[s_x, 0, t_x],
                    [0, s_y, t_y]])
    floating_im = cv2.warpAffine(floating_im, M, (n_cols, n_rows), borderMode=cv2.BORDER_REPLICATE)

    if print_debug:
        print(M)
        floating_im_copy = floating_im.copy()
        cv2.rectangle(floating_im_copy, (rect_fim.x, rect_fim.y), (rect_fim.x+rect_fim.w, rect_fim.y+rect_fim.h), color=(0, 255, 0), thickness=2)
        cv2.circle(floating_im_copy, rect_fim.center, radius=5, color=(0, 255, 0), thickness=-1)
        cv2.imshow("Traslated", floating_im_copy)
        cv2.waitKey(0)

    return floating_im


def detect_landmarks(predictor, im, dlib_rect, window_name, print_debug):
    """ Function that detects landmarks using dblib :predictor and :dblib_rect """

    if dlib_rect.width() == 0 or dlib_rect.height() == 0:
        raise RuntimeError("Invalid rectangle for landmark detection.")

    image_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    landmarks = predictor(image_gray, dlib_rect)

    im_copy = im.copy()

    points = np.int32([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)])
    corners = np.array([
        [0, 0],
        [0, im.shape[0]-1],
        [im.shape[1]-1, 0],
        [im.shape[1]-1, im.shape[0]-1],
        #[dlib_rect.left(), dlib_rect.top()],
        #[dlib_rect.right(), dlib_rect.top()],
        #[dlib_rect.left(), dlib_rect.bottom()],
        #[dlib_rect.right(), dlib_rect.bottom()]
    ])
    points = np.vstack((points, corners))

    if print_debug:
    # print(f"Points: {points}")

        for i, p in enumerate(points):
            x, y = p
            cv2.circle(im_copy, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
            cv2.putText(im_copy, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 255, 0), thickness=1)

        cv2.imshow(f"{window_name} with landmarks", im_copy)
        cv2.waitKey(0)
    return points


def compute_delaunay_triangulation(landmarks: np.ndarray, rect: tuple):
    """ Function that compute Delaunay triangulation of :landmarks """

    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks.astype(np.float32))

    # get list of triangles as [[x1,y1,x2,y2,x3,y3]]
    triangle_list = subdiv.getTriangleList()
    delaunay_triangles = []
    
    landmarks_list = landmarks.tolist()
    for t in triangle_list:
        pts = [[t[0], t[1]], [t[2], t[3]], [t[4], t[5]]]

        indices = []
        for pt in pts:
            if pt in landmarks_list:
                indices.append(landmarks_list.index(pt))
            else:
                print(f"Errore not found : {pt}")

        if len(indices) == 3:
            delaunay_triangles.append(indices) # [1, 2, 3] -> landmark1, landmark2, landmark3
        else:
            print(f"Bad triangle {indices}")

    return delaunay_triangles

def visualize_triangulation(im, landmarks, triangles, window_name):
    """ Function that visualize Delaunay triangulation of :trianles as list of indices of :landmarks """

    landmarks = landmarks.astype(int)
    im_copy = im.copy()
    for i0,i1,i2 in triangles:
        pt0 = landmarks[i0]
        pt1 = landmarks[i1]
        pt2 = landmarks[i2]

        cv2.line(im_copy, pt0, pt1, (255, 255, 255), thickness=1)
        cv2.line(im_copy, pt1, pt2, (255, 255, 255), thickness=1)
        cv2.line(im_copy, pt2, pt0, (255, 255, 255), thickness=1)

    cv2.imshow(f"{window_name} Delaunay Triangulation", im_copy)
    cv2.waitKey(0)

def transform_triangles(src_im, src_landmarks, dst_landmarks, triangles_indices):
    """ Function wrapper that calls get_triangle_mapping for each couple of triangles and remap pixels """

    h, w = src_im.shape[:2]
    map_x_full = np.full((h, w), -1, dtype=np.float32)
    map_y_full = np.full((h, w), -1, dtype=np.float32)

    for t_indices in triangles_indices:
        src_triangle = src_landmarks[t_indices].astype(np.float32)
        dst_triangle = dst_landmarks[t_indices].astype(np.float32)
        
        get_triangle_mapping(src_im, src_triangle, dst_triangle, map_x_full, map_y_full)

    transformed = cv2.remap(src_im, map_x_full, map_y_full, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Merge delle immagini laddove la maschera è presente
    result_im = np.zeros_like(src_im)
    result_im[map_x_full != -1] = transformed[map_x_full != -1]
    
    return result_im

def get_triangle_mapping(src_im, src_triangle, dst_triangle, map_x, map_y):
    """ Function that get x-y coordinates mapping from :dst_triangle to :src_triangle """

    h, w = src_im.shape[:2]
    h, w = src_im.shape[:2]

    M = cv2.getAffineTransform(dst_triangle, src_triangle)  #inverse: dst -> src
    #print(M)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_triangle.astype(int), 255)

    y_coords, x_coords = np.nonzero(mask) # np.where(mask == 255)

    homog_coords = np.vstack((x_coords, y_coords, np.ones(x_coords.shape)))
    r = M @ homog_coords

    map_x[y_coords, x_coords] = r[0, :]
    map_y[y_coords, x_coords] = r[1, :]


def create_gif(frames, gif_filename, duration=0.5):
    """ function that use imageio to create an animated ping-pong gif from :frames """

    print(f"Creating GIF: {gif_name}")
    # create the ping-pong sequence: forward + backward
    extended_frames = frames + frames[-2::-1]

    with imageio.get_writer(gif_filename, mode='I', duration=duration, loop=0) as writer:
        for frame in extended_frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_rgb.dtype != np.uint8:
                frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
            writer.append_data(frame_rgb)


def compute_analysis(fim_path, rim_path, t_max=10, print_debug=False):
    """ main function that compute face morphing """

    fim_name = "Floating im"
    rim_name = "Reference im"

    # 1. load the 2 images and ensure they have the same size
    fim = cv2.imread(fim_path)
    rim = cv2.imread(rim_path)

    assert fim is not None and rim is not None, "Error loading images"
    assert fim.shape == rim.shape, "Images have different shapes"

    height, width = rim.shape[:2]
    # 2. detect the faces and use them to align the images (translation + scale) such that the center of the faces in the two images overlap
    #   from openCV use: the faceDetector (cv2.CascadeClassifier) + transformation functions (warpAffine)
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    rect_fim = detect_face(classifier, fim, fim_name, print_debug)
    rect_rim = detect_face(classifier, rim, rim_name, print_debug)

    fim = align_faces(fim, rim, rect_fim, rect_rim, print_debug)

    # 3. detect face landmarks by dlib
    # from dlib, to detect the landmarks, use the model: dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    fim_dlib_rect = rect_fim.to_dlib_rectangle()
    rim_dlib_rect = rect_rim.to_dlib_rectangle()

    fim_landmarks = detect_landmarks(predictor, fim, fim_dlib_rect, fim_name, print_debug)
    rim_landmarks = detect_landmarks(predictor, rim, rim_dlib_rect, rim_name, print_debug)

    # 4. use openCV to compute the delaunay triangulation
    rect = (0, 0, width, height) # rect di tutta l'im
    delaunay_triangles = compute_delaunay_triangulation(rim_landmarks, rect)
    if print_debug:
        print(f"rect: {rect}")
        print(f"lndmrks: {rim_landmarks}")
        print(f"delaunay_triangles: {delaunay_triangles}")
        visualize_triangulation(fim, fim_landmarks, delaunay_triangles, fim_name)
        visualize_triangulation(rim, rim_landmarks, delaunay_triangles, rim_name)

    frames = []

    for t in np.linspace(0, 1, t_max):
        print("T: ", t)

        # 5. For varying values of t, compute intermediate landmarks and corresponding triangles
        intermediate_landmarks = (1 - t) * fim_landmarks + t * rim_landmarks
        
        # 6. use your own piecewise affine transformation function to transform both floating and reference images considering the intermediate landmarks (slide 12).
        # (useful functions: cv2.fillConvexPoly, cv2.remap, np.meshgrid.
        current_fim = transform_triangles(fim, fim_landmarks, intermediate_landmarks, delaunay_triangles)
        current_rim = transform_triangles(rim, rim_landmarks, intermediate_landmarks, delaunay_triangles)

        # 7. blend the two transformed images as detailed in slide 13
        morphed = cv2.addWeighted(current_fim, 1 - t, current_rim, t, 0.0)
        
        # 8. Collect the blended images computed for varying values of t and create a gif
        frames.append(morphed)

        if print_debug:
            cv2.imshow(f"Morph t={t:.1f}", morphed)
            vis = np.hstack((current_fim, current_rim))
            cv2.imshow("Triangle Pairs: SRC (Left) vs DST (Right)", vis)
            cv2.waitKey(0)

    create_gif(frames, gif_name, duration=0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face morphing script')
    parser.add_argument("--fim_path", type=str, default="dati/coppia_2_1.jpg", help="Floating image path")
    parser.add_argument("--rim_path", type=str, default="dati/coppia_2_2.jpg", help="Reference image path")
    parser.add_argument("--gif_name", type=str, default="result_gif", help="Path gif risultante")
    parser.add_argument("--t", type=int, default=10, help="Parametro t")
    parser.add_argument("--print_debug", action="store_true", help="Debug print")

    args = parser.parse_args()

    fim_path = args.fim_path
    rim_path = args.rim_path
    t = args.t
    print_debug = args.print_debug
    gif_name = args.gif_name
    gif_name = gif_name if ".gif" in gif_name else gif_name + ".gif"

    print("Started with following parameters:")
    print(f"\t - Floating im: {fim_path}")
    print(f"\t - Reference im: {rim_path}")
    print(f"\t - Result gif name: {gif_name}")
    print(f"\t - t: {t}")
    print(f"\t - Debug attivo: {print_debug}")

    compute_analysis(fim_path, rim_path, t, print_debug)
