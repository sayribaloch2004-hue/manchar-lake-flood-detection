import streamlit as st
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, binary_opening, binary_closing, label
from skimage.filters import threshold_otsu

st.set_page_config(page_title="Manchar Lake Flood Detection", layout="wide")

st.title("Manchar Lake Flood Detection Tool")
st.write("Upload two Sentinel-1 SAR images (before and after a flood event) to detect newly flooded areas using AI-based image analysis.")

st.sidebar.header("Upload Images")
before_upload = st.sidebar.file_uploader("Before Flood Image (.tif/.tiff)", type=["tif", "tiff"])
after_upload = st.sidebar.file_uploader("After Flood Image (.tif/.tiff)", type=["tif", "tiff"])

def get_largest_water_body(water_mask):
    cleaned = binary_opening(water_mask, structure=np.ones((3, 3)))
    cleaned = binary_closing(cleaned, structure=np.ones((15, 15)))
    labeled, num_features = label(cleaned)
    if num_features == 0:
        return cleaned
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_label = sizes.argmax()
    return labeled == largest_label

def load_image(uploaded_file):
    with rasterio.open(uploaded_file) as src:
        return src.read(1)

if before_upload is not None and after_upload is not None:
    with st.spinner("Processing images..."):
        before = load_image(before_upload)
        after = load_image(after_upload)

        before_smooth = median_filter(before, size=5)
        after_smooth = median_filter(after, size=5)

        before_thresh = threshold_otsu(before_smooth[before_smooth > 0])
        after_thresh = threshold_otsu(after_smooth[after_smooth > 0])

        before_water_raw = (before_smooth < before_thresh) & (before_smooth > 0)
        after_water_raw = (after_smooth < after_thresh) & (after_smooth > 0)

        before_lake = get_largest_water_body(before_water_raw)
        after_lake = get_largest_water_body(after_water_raw)

        combined_lake = before_lake | after_lake
        rows = np.any(combined_lake, axis=1)
        cols = np.any(combined_lake, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        buffer = 80
        rmin, rmax = max(0, rmin - buffer), min(before.shape[0], rmax + buffer)
        cmin, cmax = max(0, cmin - buffer), min(before.shape[1], cmax + buffer)

        before_lake_crop = before_lake[rmin:rmax, cmin:cmax]
        after_lake_crop = after_lake[rmin:rmax, cmin:cmax]
        flood_mask_final = after_lake_crop & (~before_lake_crop)

        total = flood_mask_final.size
        before_pixels = int(np.sum(before_lake_crop))
        after_pixels = int(np.sum(after_lake_crop))
        flood_pixels = int(np.sum(flood_mask_final))
        growth = ((after_pixels - before_pixels) / before_pixels * 100) if before_pixels > 0 else 0

    st.success("Analysis complete.")

    col1, col2, col3 = st.columns(3)
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.imshow(before_lake_crop, cmap="Blues")
        ax1.set_title("Lake Extent - Before")
        ax1.axis("off")
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots()
        ax2.imshow(after_lake_crop, cmap="Blues")
        ax2.set_title("Lake Extent - After")
        ax2.axis("off")
        st.pyplot(fig2)
    with col3:
        fig3, ax3 = plt.subplots()
        ax3.imshow(flood_mask_final, cmap="Reds")
        ax3.set_title("Newly Flooded Area")
        ax3.axis("off")
        st.pyplot(fig3)

    st.subheader("Statistics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lake area before", f"{before_pixels:,} px")
    m2.metric("Lake area after", f"{after_pixels:,} px")
    m3.metric("Newly flooded", f"{flood_pixels:,} px")
    m4.metric("Lake growth", f"{growth:.1f}%")

else:
    st.info("Upload both images in the sidebar to begin analysis.")
