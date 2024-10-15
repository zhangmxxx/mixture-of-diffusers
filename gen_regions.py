import json

regions = []

# All regions share same prompt
prompt = "Abstract decorative illustration, by lyubov popova and kadinski and kazimir malevich and mondrian, elegant, intricate, highly detailed, smooth, sharp focus, vibrant colors, artstation, stunning masterpiece"

# Create regions in an 8 * 11 grid
for i in range(8):
  for j in range(11):
    row_init = (480 - 240) * i # (tile_h - overlab_h)
    col_init = (640 - 320) * j
    region = {
      "area": [row_init, row_init + 480, col_init, col_init + 640],  # Dynamic area values
      "guidance_scale": 8,
      "prompt": prompt
    }
    regions.append(region)

# Create the final structure
output_data = {
  "regions": regions
}

# Write the output to a JSON file
with open("regions_data.json", "w") as json_file:
    json.dump(output_data, json_file, indent=4)

print("JSON file created successfully!")
