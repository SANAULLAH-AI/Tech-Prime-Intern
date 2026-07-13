
# Phase 1: Digital Image Fundamentals & Representations

---

## 1. Pixels & Dimensions (Image Basics)

For a computer, an image is not just a picture. It is a big table (grid) of numbers.

### What is a Pixel?
- **Pixel** = Picture Element
- It is the smallest dot in a digital image.
- Every pixel has its own color value.

### Coordinate System
- Image graph paper is different from math graph paper.
- The starting point `(0,0)` is at the **top-left corner**.
- When you go right → X increases
- When you go down → Y increases

### Resolution
- Total number of pixels in an image.
- Example: `1920 x 1080` means:
  - 1920 pixels wide (width)
  - 1080 pixels high (height)
  - Total = `1920 × 1080 = 2,073,600` pixels (about 2 Megapixels)

### Aspect Ratio
- Width to Height ratio
- YouTube videos = `16:9`
- Mobile screens / Instagram Reels = `9:16`

### Bit Depth
- How much color information one pixel can store.

| Bit Depth | Range | Total Colors | Use Case |
|-----------|-------|--------------|----------|
| **8-bit** | 0 to 255 | 256 shades | Normal images |
| **16-bit** | 0 to 65,535 | Thousands of shades | Professional cameras, Medical X-rays |

---

## 2. Color Spaces (World of Colors)

Computers understand colors using different rules. These rules are called **Color Spaces**.

### RGB (Red, Green, Blue)
- Most common format in the world
- Mix these 3 primary colors to make any color
- Computer screens work on RGB

### BGR
- Same as RGB but order is different
- OpenCV reads images in BGR (Blue, Green, Red)
- Not a new format, just OpenCV's default

### HSV (Hue, Saturation, Value)
- Best for human understanding
- **Hue** = What is the actual color? (Red, Yellow, Blue?)
- **Saturation** = How strong or weak is the color?
- **Value** = How bright or dark is the color?

**Why use HSV?**
- When you want to detect a specific color (like finding a red ball)
- HSV is not affected by light changes
- RGB changes when light changes, HSV stays stable

### HSL (Hue, Saturation, Lightness)
- Similar to HSV
- Small difference in how brightness is calculated

### YCrCb
- **Y** = Brightness (Luminance)
- **Cr/Cb** = Color information (Chrominance)
- Used in video compression and old TV signals

### CMYK (Cyan, Magenta, Yellow, Black)
- Only used in printing (printers)
- Not used for screens

### Grayscale
- Black and White image
- No colors, only brightness
- `0` = completely black
- `255` = completely white

### Binary Image
- Only 2 possibilities:
  - `0` = Black
  - `1` or `255` = White
- No middle value

---

## 3. How Computer Sees Images?

### 2D Matrices
- If image is Grayscale (Black & White)
- Computer sees it as a 2D table (Rows and Columns)
- Every cell has a number from `0 to 255`

**Example:**
```
[ 0  50 100 ]
[ 50 128 200 ]
[ 100 200 255 ]
```

### 3D Tensors
- If image is Colorful (RGB)
- Computer has 3 separate tables stacked together:
  - 1st table = Red values
  - 2nd table = Green values
  - 3rd table = Blue values
- This 3D structure = **Tensor**
- Dimensions = `(Height, Width, Channels)`
- RGB has `3` channels

### Image Histograms
- A graph showing:
  - Which color or brightness appears how many times
- **Left side high** = Dark image
- **Right side high** = Bright image

### Splitting and Merging Channels
- In OpenCV, you can split a colorful image into Red, Green, Blue parts
- You can also merge them back to make a colorful image

---

## 4. Image Pyramids (Small and Big Images)

When training computers to recognize objects, objects look:
- **Small** when far away
- **Big** when close

To handle this, we use **Image Pyramids**. Images are made smaller or bigger step by step.

### Gaussian Pyramid
- Each step makes the image half its size
- Images become blurry at each step
- Looks like a pyramid from top to bottom
- Used in object detection
- Helps models recognize objects of any size

### Laplacian Pyramid
- Created by taking difference between Gaussian pyramid levels
- Keeps only:
  - Fine details
  - Edges
  - Boundaries
- Used for:
  - Making images bigger again
  - Image blending (mixing images smoothly)

---

## 5. Image Formats (File Types)

### Lossy vs Lossless

| Type | What Happens? | Example |
|------|---------------|---------|
| **Lossy** | Some quality is permanently removed to make file smaller | JPEG |
| **Lossless** | File becomes smaller but quality stays 100% same | PNG |

### Common Image Formats

| Format | Description |
|--------|-------------|
| **JPEG / JPG** | Most used format on internet. Small size but pixels get damaged during compression. |
| **PNG** | Supports transparent background. Perfect quality but big file size. |
| **BMP (Bitmap)** | Raw format. No compression. Pixels stay as they are. Very large file size. |
| **TIFF** | Used in professional photography and printing. Stores very high quality. |
| **WebP** | Google's new format. PNG quality with JPEG size. Very popular on websites now. |
| **DICOM** | Special medical format. MRI and CT Scan files use this. Contains not just image but patient name, doctor details, and 3D scan data together. |

---

##  Quick Summary

| Concept | Simple Explanation |
|---------|-------------------|
| **Pixel** | Smallest dot in an image |
| **Resolution** | Width × Height = Total pixels |
| **Aspect Ratio** | Width to Height ratio |
| **Bit Depth** | How many colors a pixel can show |
| **RGB** | Most common color format for screens |
| **HSV** | Best format for color detection |
| **Grayscale** | Black and white images |
| **Tensor** | 3D structure (Height, Width, Channels) |
| **Histogram** | Graph showing pixel brightness |
| **Pyramid** | Making images smaller step by step |

---

##  Key Points to Remember

1. **Image = Grid of Numbers** - For a computer, every image is just a table of numbers.

2. **Different Color Spaces for Different Tasks**:
   - Use RGB for displaying on screen
   - Use HSV for detecting colors

3. **Higher Resolution = More Details** - More pixels mean clearer image.

4. **Pyramids Help with Size** - Models use pyramids to detect objects at different sizes.

5. **Choose Right Format**:
   - JPEG = Websites
   - PNG = Transparent backgrounds
   - TIFF = Professional work
   - DICOM = Medical images

---

> 💡 **Pro Tip**: OpenCV reads images in BGR by default. Always convert to RGB before showing with matplotlib!

---

*End of Phase 1*
```
