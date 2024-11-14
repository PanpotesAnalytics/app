import streamlit as st
from osgeo import gdal
import os
import numpy as np
from samgeo import SamGeo, show_image
import torch

# Function to normalize the band to the 8-bit range (0-255)
def normalize_band(band):
    band_min = np.min(band)
    band_max = np.max(band)
    normalized_band = ((band - band_min) / (band_max - band_min)) * 255
    return normalized_band.astype(np.uint8)

# Function to convert 8-band image to RGB
def convert_to_rgb(input_image_path, output_image_path):
    dataset = gdal.Open(input_image_path)

    # Read the bands: 6 (Red), 4 (Green), 2 (Blue)
    red_band = dataset.GetRasterBand(6).ReadAsArray()
    green_band = dataset.GetRasterBand(4).ReadAsArray()
    blue_band = dataset.GetRasterBand(2).ReadAsArray()

    # Normalize each band to the 8-bit range
    red_band_normalized = normalize_band(red_band)
    green_band_normalized = normalize_band(green_band)
    blue_band_normalized = normalize_band(blue_band)

    # Stack the normalized bands into a 3D array
    rgb_image = np.dstack((red_band_normalized, green_band_normalized, blue_band_normalized))

    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_image_path, dataset.RasterXSize, dataset.RasterYSize, 3, gdal.GDT_Byte)

    # Write bands to new dataset
    out_dataset.GetRasterBand(1).WriteArray(rgb_image[:, :, 0])
    out_dataset.GetRasterBand(2).WriteArray(rgb_image[:, :, 1])
    out_dataset.GetRasterBand(3).WriteArray(rgb_image[:, :, 2])

    out_dataset.SetProjection(dataset.GetProjection())
    out_dataset.SetGeoTransform(dataset.GetGeoTransform())

    out_dataset.FlushCache()
    out_dataset = None
    dataset = None

    st.success(f"RGB image saved to {output_image_path}")

# Function to segment image into smaller parts
def segment_image_gdal(image_path, output_folder, segment_size=512):
    dataset = gdal.Open(image_path)
    if not dataset:
        raise FileNotFoundError(f"Unable to open image at {image_path}")

    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    image_width = dataset.RasterXSize
    image_height = dataset.RasterYSize
    num_bands = dataset.RasterCount

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    segment_counter = 0

    for row_start in range(0, image_height, segment_size):
        for col_start in range(0, image_width, segment_size):
            row_end = min(row_start + segment_size, image_height)
            col_end = min(col_start + segment_size, image_width)

            segment = np.zeros((num_bands, row_end - row_start, col_end - col_start))

            for band in range(1, num_bands + 1):
                band_data = dataset.GetRasterBand(band)
                segment[band-1, :, :] = band_data.ReadAsArray(col_start, row_start, col_end-col_start, row_end-row_start)

            new_geotransform = (
                geotransform[0] + col_start * geotransform[1],
                geotransform[1],
                geotransform[2],
                geotransform[3] + row_start * geotransform[5],
                geotransform[4],
                geotransform[5]
            )

            segment_filename = os.path.join(output_folder, f"segment_{segment_counter}.tif")
            driver = gdal.GetDriverByName('GTiff')
            output_dataset = driver.Create(segment_filename, col_end-col_start, row_end-row_start, num_bands, gdal.GDT_Float32)

            for band in range(1, num_bands + 1):
                output_dataset.GetRasterBand(band).WriteArray(segment[band-1, :, :])

            output_dataset.SetGeoTransform(new_geotransform)
            output_dataset.SetProjection(projection)

            output_dataset.FlushCache()
            output_dataset = None

            segment_counter += 1

    dataset = None
    st.success(f"Image segmented into {segment_counter} sections")

# Streamlit app setup
st.title("8-Band Image Processor")

# Ensure the 'temp' directory exists
temp_dir = "temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Sidebar for selecting the process
option = st.sidebar.selectbox("Choose a process", ("Convert to RGB", "Segment Image"))

# Upload image file
uploaded_file = st.file_uploader("Drag and Drop an 8-band Image (TIF format)", type=["tif"])

if uploaded_file is not None:
    # Create a temporary location to save the uploaded file
    input_image_path = os.path.join(temp_dir, uploaded_file.name)
    
    # Save the uploaded file
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"Uploaded file: {uploaded_file.name}")

    if option == "Convert to RGB":
        # Output path input for RGB conversion
        output_image_path = st.text_input("Enter the output path (including filename) for the RGB image:", "output_rgb_image.tif")

        # Convert to RGB when the user clicks the button
        if st.button("Convert to RGB"):
            convert_to_rgb(input_image_path, output_image_path)

    elif option == "Segment Image":
        # Output path input and segment size for image segmentation
        output_folder = st.text_input("Enter the output folder for image segments:", "output_segments")
        segment_size = st.number_input("Enter segment size (default 512):", min_value=64, max_value=2048, value=2048)

        # Segment the image when the user clicks the button
        if st.button("Segment Image"):
            
            segment_image_gdal(input_image_path, output_folder, segment_size)
            b8_list = [f for f in os.listdir(output_folder)]
            for i in b8_list:
            	convert_to_rgb(output_folder+'/'+i, 'rgb_output/'+i.split('.')[0]+'rgb.tif')
            
            rgb_list = [f for f in os.listdir('rgb_output')]
            sam_kwargs = {
                "points_per_side": 32,
                "pred_iou_thresh": 0.86,
    			"stability_score_thresh": 0.92,
    			"crop_n_layers": 1,
    			"crop_n_points_downscale_factor": 2,
    			"min_mask_region_area": 100,
                }
            sam = SamGeo(
                model_type="vit_b",
                sam_kwargs=None,
                )
            for image in rgb_list:
                torch.cuda.empty_cache()
                sam.generate('rgb_output/'+image, output="segment/mask"+image, foreground=True, unique=True)
                sam.tiff_to_geojson("segment/mask"+image, "polys/"+image.split('.')[0]+'.geojson')
                torch.cuda.empty_cache()

