import os

file_path = os.path.join(
    os.path.dirname(__file__), "src", "data", "database_builder.py"
)

with open(file_path, "r") as file:
    content = file.read()

# Replace the ticker string
content = content.replace('"GC=F"', '"GLD"')

# Replace the prefix usages
content = content.replace("GC_CLOSE", "GLD_CLOSE")
content = content.replace("GC_OPEN", "GLD_OPEN")
content = content.replace("GC_HIGH", "GLD_HIGH")
content = content.replace("GC_LOW", "GLD_LOW")
content = content.replace("GC_VOLUME", "GLD_VOLUME")
content = content.replace("GC_CMF", "GLD_CMF")
content = content.replace("GC_PPO_", "GLD_PPO_")
content = content.replace("GC_HG_RATIO", "GLD_HG_RATIO")
content = content.replace("GC_HG_ZSCORE_", "GLD_HG_ZSCORE_")
content = content.replace("'GC'", "'GLD'")

# Remove GLD_VOLUME and GLD_CMF from the guillotine cols_to_drop because they are legitimate now!
content = content.replace("'GLD_VOLUME', ", "")
content = content.replace("'GLD_CMF'", "")

with open(file_path, "w") as file:
    file.write(content)

print("Successfully transformed GC to GLD in database logic!")
