from pathlib import Path
from PIL import Image



# image = Image.open('C:/Users/Allan/Documents/Epitech/T8 - Zoidberg/chest_Xray/test/NORMAL/IM-0001-0001.jpeg')
# new_image = image.resize((100, 50))
# new_image.save('C:/Users/Allan/Documents/Epitech/T8 - Zoidberg/resized/test/NORMAL/1.jpeg')


def resize(url):
    i = 0

    for txt_path in Path(f"C:/Users/Allan/Documents/Epitech/T8 - Zoidberg/chest_Xray/{url}").glob("*.jpeg"):
        image = Image.open(txt_path)
        new_image = image.resize((100, 50))
        new_image.save(f'C:/Users/Allan/Documents/Epitech/T8 - Zoidberg/resized/dataset/i-{i}.jpeg')
        i = i + 1
        
# print("test : NORMAL")
# resize("test/NORMAL/")

# print("test : PNEUMONIA")
# resize("test/PNEUMONIA/")

# print("train : NORMAL")
# resize("train/NORMAL/")

print("train : PNEUMONIA")
resize("train/PNEUMONIA/")


