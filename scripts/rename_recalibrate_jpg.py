import os
from PIL import Image

# Settings
input_folder = os.path.expanduser("~/images")
output_folder = os.path.expanduser("~/heat_seeking_missile/calib_images_resized")
target_size = (640, 640)

os.makedirs(output_folder, exist_ok=True)

# Gather supported image files
supported_exts = [".jpg", ".jpeg", ".png"]
image_files = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in supported_exts]
image_files.sort()

# Rename and resize
for idx, file_name in enumerate(image_files, start=1):
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, f"image{idx}.jpg")

    try:
        img = Image.open(input_path).convert("RGB")
        img = img.resize(target_size, Image.BILINEAR)
        img.save(output_path, "JPEG")
        print(f"✅ Saved: {output_path}")
    except Exception as e:
        print(f"❌ Error with {file_name}: {e}")

print(f"\n🎉 Finished! {len(image_files)} images saved to {output_folder}")
