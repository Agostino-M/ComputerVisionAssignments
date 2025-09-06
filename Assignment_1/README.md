# Face Morphing - 1° Assignment
> **Disclaimer:** Only the text content from the original PDF has been automatically extracted with PyMuPDF. Images have been omitted. This Markdown file is for demonstration purposes only and is **not intended to replace the original PDF**.

## Morphing

- Morphing is the process of warping and blending two or more images.  
- Cross-dissolving between two images leads to ghosting.  
- If instead we warp each image toward the other before blending, corresponding features in the two images are aligned and no ghosting results.  
- Given a set of correspondences between the two images, the alignment and blending is repeated using different blend factors and amounts of deformation at each interval.  
- This allows us to obtain a sequence of interpolated frames.  

To do face morphing we need to:
- Implement a geometric transformation to transform one image (floating image) into a transformed image that is geometrically aligned with a reference image (using a piecewise affine transformation).  
- Apply the transformation gradually by generating a sequence of intermediate transformed images that, once assembled in a video, will give the effect of the transformation from the floating to the reference images.  
- To change the face appearance, it is enough to implement a blending function as a weighted sum of the transformed images.  

---

## Geometrical Transformation

- We need to find point correspondences in the two images.  
- This can be done manually but, to make the process automatic, we can take advantage of face landmarks.  
- From the landmarks, we can get a triangle mesh. This can be done by a Delaunay Triangulation.  
- Triangles are important because we can transform the image by means of a piecewise affine transformation.  
- You can decide how many points to use to get the triangulation (the one shown is an example).  

We need to transform a triangle of the floating image into a triangle of the reference image.  
Given the corresponding triangle vertices, we can:
- Find an affine transformation.  
- Transform the pixel coordinates of one triangle into the other.  

Once this is done for each triangle pair, we can use inverse mapping to find the transformed image.  

Let’s call **Lf** the landmarks of the floating image and **Lr** the landmarks of the reference image.  
The piecewise affine transformation transforms the floating image in such a way that the landmarks of the transformed image are aligned with **Lr**.  

After applying the piecewise affine transformation by inverse mapping, we get an image whose face landmarks are aligned to those of the reference image.  

To make the transformation gradual, we can use a parameter *t* in [0, 1] to modify the landmarks to be used in the transformation:  

**LI = (1 - t) Lf + t Lr**

- When *t = 0*, LI = Lf, and the transformation returns the floating image.  
- When *t = 1*, LI = Lr, and the transformation returns the transformed image aligned with the reference image.  
- By varying *t* in [0, 1], we obtain a sequence of transformed images.  

---

## Blending

- Blending two images can be done easily by computing a weighted sum of the transformed images.  
- The weights can vary along the sequence and should sum to 1.  
- Both images must be transformed before blending.  

---

## Face Morphing – Sketch

1. Load the two images and ensure they have the same size.  
2. Detect the faces and align the images (translation + scale) so that the centers of the faces overlap.  
   - In OpenCV: use the face detector (`cv2.CascadeClassifier`) + transformation functions (`warpAffine`).  
3. Detect face landmarks using **dlib**.  
   - This step requires installing `cmake` and compiling `dlib` locally.  
   - Use the model: `dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")`.  
4. Use OpenCV to compute the Delaunay triangulation (`cv2.Subdiv2D`).  
   - You may need to add extra points to get good results.  
5. For varying values of *t*, compute intermediate landmarks and corresponding triangles.  
6. Implement your own piecewise affine transformation to transform both floating and reference images, considering the intermediate landmarks.  
   - Useful functions: `cv2.fillConvexPoly`, `cv2.remap`, `np.meshgrid`.  
   - **Do not** use functions from external libraries—implement your own version.  
7. Blend the two transformed images (careful with data types; OpenCV also has useful functions for blending).  
8. Collect the blended images for varying *t* and create a GIF (the `imageio` library is recommended since OpenCV does not support GIFs).  

---

## Results

- Result 1: Initial floating and reference images, corresponding triangles, and intermediate blending.  
- Result 2: Sequence of transformations with increasing *t*.  
- Result 3: Final morphing showing smooth transition between the two faces.  
